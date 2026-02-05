import bcrypt
import boto3
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import requests

from datetime import datetime, timedelta
from pathlib import Path

def load_ndjson_data(file_path):
    """Load NDJSON (New-line Delimited JSON) data into a list."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def check_password(tab_key="default"):
    """Returns True if user is authenticated, False otherwise. Shows password input if not authenticated."""
    # Get timeout duration from secrets (default 30 minutes)
    timeout_minutes = int(st.secrets.get("AUTH_TIMEOUT_MINUTES", "30"))
    
    # Initialize session state for authentication per tab
    auth_key = f'authenticated_{tab_key}'
    timestamp_key = f'auth_timestamp_{tab_key}'
    
    if auth_key not in st.session_state:
        st.session_state[auth_key] = False
        st.session_state[timestamp_key] = None
    
    # Check if authenticated and session hasn't expired for this specific tab
    if st.session_state[auth_key]:
        if st.session_state[timestamp_key]:
            elapsed = datetime.now() - st.session_state[timestamp_key]
            if elapsed > timedelta(minutes=timeout_minutes):
                # Session expired - reset authentication for this tab
                st.session_state[auth_key] = False
                st.session_state[timestamp_key] = None
                st.warning("Session expired. Please login again.")
            else:
                # Session valid - refresh timestamp for idle timeout
                st.session_state[timestamp_key] = datetime.now()
                return True
    
    # Show password input
    st.warning("This content is password protected.")
    password_input = st.text_input("Enter password to access:", type="password", key=f"password_input_{tab_key}")
    
    if st.button("Submit", key=f"password_submit_{tab_key}"):
        # Get hashed password from secrets based on tab
        password_key = f"PASSWORD_HASH_{tab_key.upper()}"
        stored_hash = st.secrets.get(password_key, "").encode('utf-8')
        
        if stored_hash and bcrypt.checkpw(password_input.encode('utf-8'), stored_hash):
            st.session_state[auth_key] = True
            st.session_state[timestamp_key] = datetime.now()
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")
    
    return False

@st.cache_data
def load_from_s3(file_key):
    """Load NDJSON data from S3 bucket. Cache refreshes daily based on date."""
    cache_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Get AWS credentials from Streamlit secrets
        s3_client = boto3.client(
            's3',
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_REGION"],
            endpoint_url=st.secrets["S3_ENDPOINT"]
        )
        
        # Download file from S3
        bucket = st.secrets["BUCKET"]
        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        
        # Parse NDJSON content
        raw_data = [json.loads(line) for line in content.splitlines() if line.strip()]
        return raw_data, None
        
    except Exception as e:
        return None, str(e)

@st.cache_data
def get_gbif_info(taxon_id):
    """Fetch taxonomic information from GBIF API for a given taxon ID."""
    try:
        url = f"https://api.gbif.org/v1/species/{taxon_id}"
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

def extract_labels(data):
    """Extract relevant label information from raw data."""
    labels = []
    images = []
    
    for item in data:
        # Get base information
        data_row = item['data_row']
        projects = item['projects']
        
        # Add image information regardless of label status
        image_info = {
            'image_id': data_row['global_key'],
            'status': 'Not labeled'  # Default status
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
                                        'taxon': obj['classifications'][0]['checklist_answers'][0]['name'],
                                        'taxon_id': obj['classifications'][0]['checklist_answers'][0]['value'],
                                        'status': 'Labeled'
                                    }
                                    labels.append(annotation)
                            except (IndexError, KeyError) as e:
                                continue
        
        images.append(image_info)
    
    # Create DataFrames
    labels_df = pd.DataFrame(labels) if labels else pd.DataFrame()
    images_df = pd.DataFrame(images)
    
    # Add taxonomic rank information if we have labels
    if not labels_df.empty:
        info_placeholder = st.empty()
        info_placeholder.info("Fetching taxonomic information from GBIF API...")
        unique_taxon_ids = labels_df['taxon_id'].unique()
        
        # Create a progress bar
        progress_bar = st.progress(0)
        gbif_info_mapping = {}
        
        for i, taxon_id in enumerate(unique_taxon_ids):
            gbif_info_mapping[taxon_id] = get_gbif_info(taxon_id)
            progress_bar.progress((i + 1) / len(unique_taxon_ids))
        
        # Clear the progress bar and info message
        progress_bar.empty()
        info_placeholder.empty()
        
        # Map ranks and GBIF info to labels
        labels_df['taxonomic_rank'] = labels_df['taxon_id'].map(lambda x: gbif_info_mapping[x]['rank'])
        labels_df['gbif_species'] = labels_df['taxon_id'].map(lambda x: gbif_info_mapping[x]['species'])
        labels_df['gbif_genus'] = labels_df['taxon_id'].map(lambda x: gbif_info_mapping[x]['genus'])
        labels_df['gbif_family'] = labels_df['taxon_id'].map(lambda x: gbif_info_mapping[x]['family'])
        
        # Create gbif_taxon field based on rank
        def get_gbif_taxon(row):
            rank = row['taxonomic_rank']
            if rank == 'SPECIES':
                return row['gbif_species']
            elif rank == 'GENUS':
                return row['gbif_genus']
            elif rank == 'FAMILY':
                return row['gbif_family']
            else:
                return row['gbif_genus']  # Default to genus for other ranks
        
        labels_df['gbif_taxon'] = labels_df.apply(get_gbif_taxon, axis=1)
        
        return labels_df, images_df, gbif_info_mapping
    
    return labels_df, images_df, {}

def process_and_display_data(all_labels, all_images, tab_key):
    """Process and display label and image data."""

    # Process images data
    if all_images:
        images_df = pd.concat(all_images, ignore_index=True)
        
        # Add status filter
        st.header("Filter Options")
        col1, col2, col3 = st.columns([1, 1, 2])  # Make left column smaller
        with col1:
            # Create status options with counts
            status_counts = images_df['status'].value_counts()
            status_options = [f"ALL ({len(images_df)})"]
            status_mapping = {'ALL': f"ALL ({len(images_df)})"}
            
            for status in sorted(images_df['status'].unique().tolist()):
                count = status_counts[status]
                display_text = f"{status} ({count})"
                status_options.append(display_text)
                status_mapping[status] = display_text
            
            st.subheader("Filter by status")
            selected_status_display = st.selectbox("Labelbox status", status_options, index=0, key=f"status_filter_{tab_key}")
            
            # Extract actual status from display text
            selected_status = selected_status_display.split(' (')[0]
        
        with col2:
            # Add taxonomic rank filter for labels
            if all_labels:
                rank_df = pd.concat(all_labels, ignore_index=True)
                
                if not rank_df.empty and 'taxonomic_rank' in rank_df.columns:
                    st.subheader("Filter by rank")
                    
                    # Create rank options with counts, merging SUBSPECIES and VARIETY with SPECIES
                    rank_counts = rank_df['taxonomic_rank'].value_counts()
                    
                    # Combine SUBSPECIES and VARIETY with SPECIES
                    species_count = (rank_counts.get('SPECIES', 0) + 
                                   rank_counts.get('SUBSPECIES', 0) + 
                                   rank_counts.get('VARIETY', 0))
                    
                    rank_options = [f"ALL ({len(rank_df)})"]
                    
                    # Get unique ranks excluding SUBSPECIES and VARIETY
                    unique_ranks = [r for r in sorted(rank_df['taxonomic_rank'].unique().tolist()) 
                                  if r not in ['SUBSPECIES', 'VARIETY']]
                    
                    for rank in unique_ranks:
                        if rank == 'SPECIES':
                            # Use combined count for SPECIES
                            display_text = f"SPECIES ({species_count})"
                        else:
                            count = rank_counts[rank]
                            display_text = f"{rank} ({count})"
                        rank_options.append(display_text)
                    
                    selected_rank_display = st.selectbox(
                        "Taxonomic rank",
                        rank_options,
                        index=0,
                        key=f"rank_filter_{tab_key}"
                    )
                    
                    # Extract actual rank from display text
                    selected_rank = selected_rank_display.split(' (')[0]

        with col3:
            # Add date filter for labels
            if all_labels:
                date_df = pd.concat(all_labels, ignore_index=True)
                date_df['created_at'] = pd.to_datetime(date_df['created_at'], errors='coerce')
                date_df = date_df.dropna(subset=['created_at'])
                
                if not date_df.empty:
                    st.subheader("Filter by date")
                    
                    min_date = date_df['created_at'].min().date()
                    max_date = date_df['created_at'].max().date()

                    # Reset button
                    if st.button("Reset to full date range", key=f"reset_date_{tab_key}"):
                        st.session_state[f"start_date_{tab_key}"] = min_date
                        st.session_state[f"end_date_{tab_key}"] = max_date

                    col1, col2 = st.columns(2)
                    with col1:
                        start_date = st.date_input(
                            "Start date",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key=f"start_date_{tab_key}"
                        )
                    
                    with col2:
                        end_date = st.date_input(
                            "End date",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key=f"end_date_{tab_key}"
                        )
        
        # Filter data based on selection
        if selected_status != 'ALL':
            filtered_images_df = images_df[images_df['status'] == selected_status]
        else:
            filtered_images_df = images_df

        # Show status distribution if 'ALL' is selected
        if selected_status == 'ALL':
            st.subheader("Image Status Distribution")
            col1, col2 = st.columns([1, 1])
            with col1:
                status_counts_images = filtered_images_df['status'].value_counts()
                fig = px.pie(values=status_counts_images.values, names=status_counts_images.index)
                st.plotly_chart(fig)

    # Process labels data
    if all_labels and selected_status != 'TO_LABEL':
        df = pd.concat(all_labels, ignore_index=True).copy()
        
        # Apply the same filter from images if available
        if all_images:
            filtered_image_ids = filtered_images_df['image_id'].unique()
            df = df[df['image_id'].isin(filtered_image_ids)].copy()
        
        # Apply date filter
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df = df.dropna(subset=['created_at']).copy()
        
        if not df.empty and 'start_date' in locals() and 'end_date' in locals():
            df = df[
                (df['created_at'].dt.date >= start_date) & 
                (df['created_at'].dt.date <= end_date)
            ].copy()
        
        # Apply taxonomic rank filter
        if 'selected_rank' in locals() and selected_rank != 'ALL':
            if selected_rank == 'SPECIES':
                # Include SPECIES, SUBSPECIES, and VARIETY
                df = df[df['taxonomic_rank'].isin(['SPECIES', 'SUBSPECIES', 'VARIETY'])].copy()
            else:
                df = df[df['taxonomic_rank'] == selected_rank].copy()
        
        st.subheader("Label Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Images labeled", len(df['image_id'].unique()))
        with col2:
            st.metric("Total labels", len(df))
        with col3:
            images_with_multiple_labels = df['image_id'].value_counts()
            num_images_with_multiple_labels = (images_with_multiple_labels > 1).sum()
            st.metric("Images with >1 label", num_images_with_multiple_labels)
        with col4:
            st.metric("Number of labelers", df['labeler'].nunique())
        
        # Labels visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Labels by Labeler")
            labeler_counts = df['labeler'].value_counts()
            fig = px.pie(values=labeler_counts.values, names=labeler_counts.index)
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Labels by Taxonomic Rank")
            rank_counts = df['taxonomic_rank'].value_counts()
            fig = px.pie(values=rank_counts.values, names=rank_counts.index)
            st.plotly_chart(fig)
        
        st.subheader("Labels by Taxon")
        
        # Add user-configurable limit for number of taxa to display
        col1, col2 = st.columns([1, 3])
        with col1:
            max_taxa_display = st.slider(
                "Number of taxa to display",
                min_value=5,
                max_value=min(400, len(df['gbif_taxon'].unique())),
                value=min(75, len(df['gbif_taxon'].unique())),
                step=5,
                key=f"max_taxa_{tab_key}"
            )
        
        # Count by gbif_taxon
        taxon_counts = df['gbif_taxon'].value_counts()
        # Keep only the top N taxa by count (user-configurable)
        taxon_counts = taxon_counts.head(max_taxa_display)
        
        chart_orientation = st.radio(
            "Chart orientation",
            ["Vertical", "Horizontal"],
            horizontal=True,
            key=f"taxon_chart_orientation_{tab_key}"
        )
        
        if chart_orientation == "Horizontal":
            fig = px.bar(
                x=taxon_counts.values,
                y=taxon_counts.index,
                orientation='h',
                height=max(400, len(taxon_counts) * 20)
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Count",
                yaxis_title="Taxon"
            )
        else:
            fig = px.bar(
                x=taxon_counts.index,
                y=taxon_counts.values,
                height=500
            )
            fig.update_layout(
                xaxis={'categoryorder': 'total descending'},
                xaxis_title="Taxon",
                yaxis_title="Count"
            )
            fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig)


        # with col2:
        #                 # st.subheader("Labels by Taxonomic Rank")
        #     # rank_counts = df['taxonomic_rank'].value_counts()
        #     # fig = px.bar(
        #     #     x=rank_counts.index,
        #     #     y=rank_counts.values,
        #     #     title="Distribution of Taxonomic Ranks"
        #     # )
        #     # fig.update_layout(
        #     #     xaxis_title="Taxonomic Rank",
        #     #     yaxis_title="Count"
        #     # )
        #     # st.plotly_chart(fig)
        #     st.subheader("taxon Count by Rank")
        #     rank_taxon_counts = df.groupby('taxonomic_rank')['taxon_id'].nunique().reset_index()
        #     fig = px.pie(
        #         rank_taxon_counts,
        #         values='taxon_id',
        #         names='taxonomic_rank',
        #         title="Unique taxon Distribution by Rank"
        #     )
        #     st.plotly_chart(fig)
        
        # # Detailed breakdown table
        # st.subheader("Detailed Rank Breakdown")
        # rank_summary = df.groupby('taxonomic_rank').agg(
        #     Unique_taxon=('taxon_id', 'nunique'),
        #     Total_Labels=('taxon_id', 'count')
        # ).reset_index()
        # rank_summary.rename(columns={
        #     'taxonomic_rank': 'Taxonomic Rank',
        #     'Unique_taxon': 'Unique taxon',
        #     'Total_Labels': 'Total Labels'
        # }, inplace=True)
        # st.dataframe(rank_summary)
        
        # Show taxon list with rank information
        st.subheader("Taxa List")
        
        # Create comprehensive taxa DataFrame - group by taxon_id and get first taxon name
        taxa_with_ranks = df.groupby(['taxon_id', 'taxonomic_rank']).agg(
            taxon=('taxon', 'first'),
            label_count=('taxon_id', 'count')
        ).reset_index()
        taxa_with_ranks = taxa_with_ranks[['taxon', 'taxon_id', 'taxonomic_rank', 'label_count']]
        taxa_with_ranks = taxa_with_ranks.sort_values(['taxonomic_rank', 'taxon'])
        
        # Display the table
        st.dataframe(taxa_with_ranks)
        
        # Convert DataFrame to CSV for download
        csv = taxa_with_ranks.to_csv(index=False).encode('utf-8')
        
        # Add download button
        st.download_button(
            label="Download Taxa List with Ranks as CSV",
            data=csv,
            file_name=f"{tab_key}_taxa_list.csv",
            mime="text/csv",
            key=f"download_taxa_ranks_{tab_key}"
        )
        
        # New section for species-level labels with image URLs
        st.subheader("Species-Level Labels with Image Data")
        
        # Filter for species rank only
        species_labels = df[df['taxonomic_rank'] == 'SPECIES'].copy()
        
        if not species_labels.empty:
            # Create summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Species Labels", len(species_labels))
            with col2:
                st.metric("Unique Species", species_labels['taxon'].nunique())
            
            # Create download data with image URLs
            species_download_data = species_labels[['image_id', 'taxon', 'taxon_id', 'labeler', 'created_at']].copy()
            
            # Rename columns for clarity
            species_download_data.rename(columns={
                'image_id': 'Image_ID',
                'taxon': 'Species_Name',
                'taxon_id': 'GBIF_Taxon_ID',
                'labeler': 'Labeled_By',
                'created_at': 'Label_Date'
            }, inplace=True)
            
            # Display preview
            st.write(f"Preview of species labels ({len(species_download_data)} records):")
            st.dataframe(species_download_data.head(10))
            
            # Convert to CSV for download
            species_csv = species_download_data.to_csv(index=False).encode('utf-8')
            
            # Download button for species data
            st.download_button(
                label="Download Species Labels with Image URLs as CSV",
                data=species_csv,
                file_name=f"{tab_key}_species_images.csv",
                mime="text/csv",
                help="Download all labels with SPECIES rank including image URLs",
                key=f"download_species_{tab_key}"
            )
        else:
            st.info("No species-level labels found in the dataset.")
    
        # Update rank_summary table to include only total counts for unique species, genera, families, and distinct taxon IDs
        rank_summary = pd.DataFrame({
            'Category': ['Unique Species', 'Unique Genera', 'Unique Families', 'Distinct Taxon IDs'],
            'Total Count': [
                df['gbif_species'].nunique(),
                df['gbif_genus'].nunique(),
                df['gbif_family'].nunique(),
                df['taxon_id'].nunique()
            ]
        })

        # Display updated table
        st.dataframe(rank_summary)
        
        # New section for label counts at species, genus, and family levels
        st.subheader("Label Counts by Taxonomic Levels")

        # Count labels at each level
        taxonomic_level_counts = df['taxonomic_rank'].value_counts()

        # Create a DataFrame for display
        taxonomic_level_summary = pd.DataFrame({
            'Taxonomic Level': taxonomic_level_counts.index,
            'Label Count': taxonomic_level_counts.values
        })

        # Display the table
        st.dataframe(taxonomic_level_summary)
        
    if not all_labels and not all_images:
        st.error("No valid data found")

def main():
    st.set_page_config(page_title="Labelbox Dashboard", layout="wide")
    st.title("Labelbox Dashboard")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Upload Files", "Barro Colorado Island", "Tiputini Biodiversity Station"])
    
    # Tab 1: File Upload
    with tab1:
        uploaded_files = st.file_uploader(
            "Choose NDJSON file(s)", 
            type="ndjson",
            accept_multiple_files=True
        )

        if not uploaded_files:
            st.warning("Please upload one or more NDJSON files exported from Labelbox to begin.")
        else:
            all_labels = []
            all_images = []
            
            for uploaded_file in uploaded_files:
                st.write(f"Processing: {uploaded_file.name}")
                try:
                    content = uploaded_file.getvalue().decode()
                    raw_data = [json.loads(line) for line in content.splitlines() if line.strip()]
                    
                    labels_df, images_df, gbif_info_mapping = extract_labels(raw_data)
                    
                    if not images_df.empty:
                        all_images.append(images_df)
                    if not labels_df.empty:
                        all_labels.append(labels_df)
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue
            
            # Display the data
            process_and_display_data(all_labels, all_images, 'upload')
    
    # Tab 2: Barro Colorado Island
    with tab2:
        # Check authentication first
        if check_password("bci"):       
            info_placeholder = st.empty()
            info_placeholder.info("Loading data from Arbutus...")
            raw_data, error = load_from_s3('2024_BCI.json')
            info_placeholder.empty()
            
            if error:
                st.error(f"Error loading BCI data: {error}")
            elif raw_data:
                all_labels = []
                all_images = []
                
                try:
                    labels_df, images_df, gbif_info_mapping = extract_labels(raw_data)
                    
                    if not images_df.empty:
                        all_images.append(images_df)
                    if not labels_df.empty:
                        all_labels.append(labels_df)
                    
                    # Display the data
                    process_and_display_data(all_labels, all_images, '2024_BCI')
                    
                except Exception as e:
                    st.error(f"Error processing BCI data: {str(e)}")
            else:
                st.warning("No data found for Barro Colorado Island")
    
    # Tab 3: Tiputini Biodiversity Station
    with tab3:
        # Check authentication first
        if check_password("tbs"):
            info_placeholder = st.empty()
            info_placeholder.info("Loading data from Arbutus...")
            raw_data, error = load_from_s3('2025_TBS.json')
            info_placeholder.empty()
            
            if error:
                st.error(f"Error loading TBS data: {error}")
            elif raw_data:
                all_labels = []
                all_images = []
                
                try:
                    labels_df, images_df, gbif_info_mapping = extract_labels(raw_data)
                    
                    if not images_df.empty:
                        all_images.append(images_df)
                    if not labels_df.empty:
                        all_labels.append(labels_df)
                    
                    # Display the data
                    process_and_display_data(all_labels, all_images, '2025_TBS')
                    
                except Exception as e:
                    st.error(f"Error processing TBS data: {str(e)}")
            else:
                st.warning("No data found for Tiputini Biodiversity Station")

if __name__ == "__main__":
    main()