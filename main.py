import json
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

def load_ndjson_data(file_path):
    """Load NDJSON (New-line Delimited JSON) data into a list."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

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
    
    return annotations_df, images_df

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
    
    for uploaded_file in uploaded_files:
        st.write(f"Processing: {uploaded_file.name}")
        try:
            content = uploaded_file.getvalue().decode()
            raw_data = [json.loads(line) for line in content.splitlines() if line.strip()]
            
            annotations_df, images_df = extract_annotations(raw_data)
            
            if not images_df.empty:
                all_images.append(images_df)
            if not annotations_df.empty:
                all_annotations.append(annotations_df)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue

    # Process images data
    if all_images:
        images_df = pd.concat(all_images, ignore_index=True)
        
        # Dashboard metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Images", len(images_df))
        with col2:
            status_counts = images_df['status'].value_counts()
            st.metric("Images Annotated", 
                     status_counts.get('DONE', 0) + status_counts.get('IN_REVIEW', 0))

        # Show status distribution
        st.subheader("Image Status Distribution")
        fig = px.pie(values=status_counts.values, names=status_counts.index)
        st.plotly_chart(fig)

    # Process annotations data
    if all_annotations:
        df = pd.concat(all_annotations, ignore_index=True)
        
        st.subheader("Annotation Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Annotations", len(df))
        with col2:
            st.metric("Unique Taxa", df['taxa'].nunique())
        with col3:
            st.metric("Number of Labelers", df['labeler'].nunique())
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Annotations by Taxa")
            taxa_counts = df['taxa'].value_counts()
            
            chart_orientation = st.radio(
                "Chart orientation",
                ["Horizontal", "Vertical"],
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
        
        # Show taxa list with download button
        st.subheader("Taxa List")
        taxa_list = sorted(df['taxa'].unique())

        # Create DataFrame for export
        taxa_df = pd.DataFrame({'Taxa': taxa_list})
        
        # Convert DataFrame to CSV
        csv = taxa_df.to_csv(index=False).encode('utf-8')
        
        # Add download button
        st.download_button(
            label="Download Taxa List as CSV",
            data=csv,
            file_name="taxa_list.csv",
            mime="text/csv"
        )
        
        # Display the list in the dashboard
        st.write(taxa_list)
    
    if not all_annotations and not all_images:
        st.error("No valid data found in uploaded files")

if __name__ == "__main__":
    main()