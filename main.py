import json
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from pathlib import Path

def load_ndjson_data(file_path):
    """Load NDJSON (New-line Delimited JSON) data into a list."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

@st.cache_data
def get_gbif_info(taxa_id):
    """Fetch taxonomic information from GBIF API for a given taxon ID."""
    try:
        url = f"https://api.gbif.org/v1/species/{taxa_id}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {
                'rank': data.get('rank', 'Unknown'),
                'species': data.get('species', 'Unknown'),
                'genus': data.get('genus', 'Unknown'),
                'family': data.get('family', 'Unknown')
            }
        else:
            return {'rank': 'Unknown', 'species': 'Unknown', 'genus': 'Unknown', 'family': 'Unknown'}
    except Exception as e:
        return {'rank': 'Unknown', 'species': 'Unknown', 'genus': 'Unknown', 'family': 'Unknown'}

def extract_annotations(data):
    """Extract relevant annotation information from raw data."""
    annotations = []
    images = []
    
    for item in data:
        # Get base information
        data_row = item['data_row']
        projects = item['projects']
        
        # Add image information regardless of annotation status
        image_info = {
            'image_id': data_row['global_key'],
            'status': 'Not annotated'  # Default status
        }
        
        for project_id, project_info in projects.items():
            image_info['project_name'] = project_info['name']
            
            # Update status based on workflow_status
            if 'project_details' in project_info:
                image_info['status'] = project_info['project_details'].get('workflow_status', 'Unknown')
            
            # Process labels if they exist
            if 'labels' in project_info and project_info['labels']:
                for label in project_info['labels']:
                    if ('annotations' in label and 
                        'objects' in label['annotations'] and 
                        label['annotations']['objects']):
                        
                        for obj in label['annotations']['objects']:
                            try:
                                if ('classifications' in obj and 
                                    obj['classifications'] and 
                                    'checklist_answers' in obj['classifications'][0]):
                                    
                                    annotation = {
                                        'image_id': data_row['global_key'],
                                        'project_name': project_info['name'],
                                        'labeler': label['label_details'].get('created_by', 'Unknown'),
                                        'created_at': label['label_details'].get('created_at', ''),
                                        'taxa': obj['classifications'][0]['checklist_answers'][0]['name'],
                                        'taxa_id': obj['classifications'][0]['checklist_answers'][0]['value'],
                                        'status': 'Annotated'
                                    }
                                    annotations.append(annotation)
                            except (IndexError, KeyError) as e:
                                continue
        
        images.append(image_info)
    
    # Create DataFrames
    annotations_df = pd.DataFrame(annotations) if annotations else pd.DataFrame()
    images_df = pd.DataFrame(images)
    
    # Add taxonomic rank information if we have annotations
    if not annotations_df.empty:
        info_placeholder = st.empty()
        info_placeholder.info("Fetching taxonomic information from GBIF API...")
        unique_taxa_ids = annotations_df['taxa_id'].unique()
        
        # Create a progress bar
        progress_bar = st.progress(0)
        gbif_info_mapping = {}
        
        for i, taxa_id in enumerate(unique_taxa_ids):
            gbif_info_mapping[taxa_id] = get_gbif_info(taxa_id)
            progress_bar.progress((i + 1) / len(unique_taxa_ids))
        
        # Clear the progress bar and info message
        progress_bar.empty()
        info_placeholder.empty()
        
        # Map ranks to annotations
        annotations_df['taxonomic_rank'] = annotations_df['taxa_id'].map(lambda x: gbif_info_mapping[x]['rank'])
        
        return annotations_df, images_df, gbif_info_mapping
    
    return annotations_df, images_df, {}

def main():
    st.set_page_config(page_title="Labelbox Annotations Dashboard", layout="wide")
    st.title("Labelbox Annotations Dashboard")

    uploaded_files = st.file_uploader(
        "Choose NDJSON file(s)", 
        type="ndjson",
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.warning("Please upload one or more NDJSON files to begin.")
        return

    all_annotations = []
    all_images = []
    all_gbif_info = {}
    
    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")
        try:
            content = uploaded_file.getvalue().decode()
            raw_data = [json.loads(line) for line in content.splitlines() if line.strip()]
            
            annotations_df, images_df, gbif_info_mapping = extract_annotations(raw_data)
            
            if not images_df.empty:
                all_images.append(images_df)
            if not annotations_df.empty:
                all_annotations.append(annotations_df)
            # Merge GBIF info mappings from all files
            all_gbif_info.update(gbif_info_mapping)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue

    # Process images data
    if all_images:
        images_df = pd.concat(all_images, ignore_index=True)
        
        # Add status filter
        st.header("Filter Options")
        col1, col2 = st.columns([1, 3])  # Make left column smaller
        with col1:
            available_statuses = ['All'] + sorted(images_df['status'].unique().tolist())
            selected_status = st.selectbox("Filter by Status", available_statuses, index=0)
        
        # Filter data based on selection
        if selected_status != 'All':
            filtered_images_df = images_df[images_df['status'] == selected_status]
        else:
            filtered_images_df = images_df
        
        # Dashboard metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Images", len(images_df))
        with col2:
            st.metric("Images Annotated", len(pd.concat(all_annotations, ignore_index=True)['image_id'].unique()))

        # Show status distribution
        st.subheader("Image Status Distribution")
        status_counts_images = images_df['status'].value_counts()
        fig = px.pie(values=status_counts_images.values, names=status_counts_images.index)
        st.plotly_chart(fig)

    # Process annotations data
    if all_annotations:
        df = pd.concat(all_annotations, ignore_index=True)
        
        st.subheader("Annotation Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Annotations", len(df))
        with col2:
            st.metric("Unique Taxa", df['taxa'].nunique())
        with col3:
            st.metric("Number of Labelers", df['labeler'].nunique())
        with col4:
            images_with_multiple_annotations = df['image_id'].value_counts()
            num_images_with_multiple_annotations = (images_with_multiple_annotations > 1).sum()
            st.metric("Images with >1 Annotation", num_images_with_multiple_annotations)
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Annotations by Taxa")
            taxa_counts = df['taxa'].value_counts()
            
            chart_orientation = st.radio(
                "Chart orientation",
                ["Vertical", "Horizontal"],
                horizontal=True,
                key="taxa_chart_orientation"
            )
            
            if chart_orientation == "Horizontal":
                fig = px.bar(
                    x=taxa_counts.values,
                    y=taxa_counts.index,
                    orientation='h',
                    height=max(400, len(taxa_counts) * 20)
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis_title="Count",
                    yaxis_title="Taxa"
                )
            else:
                fig = px.bar(
                    x=taxa_counts.index,
                    y=taxa_counts.values,
                    height=500
                )
                fig.update_layout(
                    xaxis={'categoryorder': 'total descending'},
                    xaxis_title="Taxa",
                    yaxis_title="Count"
                )
                fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Annotations by Labeler")
            labeler_counts = df['labeler'].value_counts()
            fig = px.pie(values=labeler_counts.values, names=labeler_counts.index)
            st.plotly_chart(fig)
        
        # New visualizations for taxonomic ranks
        st.subheader("Taxonomic Rank Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Annotations by Taxonomic Rank")
            rank_counts = df['taxonomic_rank'].value_counts()
            fig = px.bar(
                x=rank_counts.index,
                y=rank_counts.values,
                title="Distribution of Taxonomic Ranks"
            )
            fig.update_layout(
                xaxis_title="Taxonomic Rank",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Taxa Count by Rank")
            rank_taxa_counts = df.groupby('taxonomic_rank')['taxa'].nunique().reset_index()
            fig = px.pie(
                rank_taxa_counts,
                values='taxa',
                names='taxonomic_rank',
                title="Unique Taxa Distribution by Rank"
            )
            st.plotly_chart(fig)
        
        # Detailed breakdown table
        st.subheader("Detailed Rank Breakdown")
        rank_summary = df.groupby('taxonomic_rank').agg({
            'taxa': 'nunique',
            'taxa_id': 'count',
        }).rename(columns={
            'taxa': 'Unique Taxa',
            'taxa_id': 'Total Annotations',
        }).reset_index()
        rank_summary.rename(columns={'taxonomic_rank': 'Taxonomic Rank'}, inplace=True)
        st.dataframe(rank_summary, use_container_width=True)
        
        # Show taxa list with rank information
        st.subheader("Taxa List with Taxonomic Ranks")
        
        # Create comprehensive taxa DataFrame
        taxa_with_ranks = df.groupby(['taxa', 'taxa_id', 'taxonomic_rank']).size().reset_index(name='annotation_count')
        taxa_with_ranks = taxa_with_ranks.sort_values(['taxonomic_rank', 'taxa'])
        
        # Display the table
        st.dataframe(taxa_with_ranks, use_container_width=True)
        
        # Convert DataFrame to CSV for download
        csv = taxa_with_ranks.to_csv(index=False).encode('utf-8')
        
        # Add download button
        st.download_button(
            label="Download Taxa List with Ranks as CSV",
            data=csv,
            file_name="taxa_list_with_ranks.csv",
            mime="text/csv"
        )
        
        # New section for species-level annotations with image URLs
        st.subheader("Species-Level Annotations with Image Data")
        
        # Filter for species rank only
        species_annotations = df[df['taxonomic_rank'] == 'SPECIES'].copy()
        
        if not species_annotations.empty:
            # Create summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Species Annotations", len(species_annotations))
            with col2:
                st.metric("Unique Species", species_annotations['taxa'].nunique())
            
            # Create download data with image URLs
            # Note: Assuming image_id can be used to construct URLs or you have URL data
            species_download_data = species_annotations[['image_id', 'taxa', 'taxa_id', 'labeler', 'created_at']].copy()
            
            # Rename columns for clarity
            species_download_data.rename(columns={
                'image_id': 'Image_ID',
                'taxa': 'Species_Name',
                'taxa_id': 'GBIF_Taxon_ID',
                'labeler': 'Annotated_By',
                'created_at': 'Annotation_Date'
            }, inplace=True)
            
            # Display preview
            st.write(f"Preview of species annotations ({len(species_download_data)} records):")
            st.dataframe(species_download_data.head(10), use_container_width=True)
            
            # Convert to CSV for download
            species_csv = species_download_data.to_csv(index=False).encode('utf-8')
            
            # Download button for species data
            st.download_button(
                label="Download Species Annotations with Image URLs as CSV",
                data=species_csv,
                file_name="species_annotations_with_images.csv",
                mime="text/csv",
                help="Download all annotations with SPECIES rank including image URLs"
            )
        else:
            st.info("No species-level annotations found in the dataset.")
    
        # Map GBIF information to the main DataFrame
        df['GBIF_Species'] = df['taxa_id'].map(lambda x: all_gbif_info.get(x, {}).get('species', 'Unknown'))
        df['GBIF_Genus'] = df['taxa_id'].map(lambda x: all_gbif_info.get(x, {}).get('genus', 'Unknown'))
        df['GBIF_Family'] = df['taxa_id'].map(lambda x: all_gbif_info.get(x, {}).get('family', 'Unknown'))

        # Update rank_summary table to include only total counts for unique species, genera, families, and distinct taxa IDs
        rank_summary = pd.DataFrame({
            'Category': ['Unique Species', 'Unique Genera', 'Unique Families', 'Distinct Taxa IDs'],
            'Total Count': [
                df['GBIF_Species'].nunique(),
                df['GBIF_Genus'].nunique(),
                df['GBIF_Family'].nunique(),
                df['taxa_id'].nunique()
            ]
        })

        # Display updated table
        st.dataframe(rank_summary, use_container_width=True)
        
        # New section for annotation counts at species, genus, and family levels
        st.subheader("Annotation Counts by Taxonomic Levels")

        # Count annotations at each level
        taxonomic_level_counts = df['taxonomic_rank'].value_counts()

        # Create a DataFrame for display
        taxonomic_level_summary = pd.DataFrame({
            'Taxonomic Level': taxonomic_level_counts.index,
            'Annotation Count': taxonomic_level_counts.values
        })

        # Display the table
        st.dataframe(taxonomic_level_summary, use_container_width=True)
        
    if not all_annotations and not all_images:
        st.error("No valid data found in uploaded files")

if __name__ == "__main__":
    main()