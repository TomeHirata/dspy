import streamlit as st
import pandas as pd
import numpy as np
import dspy
import os
import json

import dspy.evaluate
import dspy.evaluate.metrics
import dspy.signatures


# Set page title and configuration
st.set_page_config(page_title="Streamlit Data Explorer", layout="wide")

# Header section
st.title("Data Explorer Dashboard")
st.markdown("Upload your CSV file or use sample data to optimize propmts.")

# Function to generate sample data
@st.cache_data
def generate_data(size):
    dates = pd.date_range(start="2023-01-01", periods=size)
    values = np.cumsum(np.random.randn(size)) + 100
    noise = np.random.randn(size) * 5
    df = pd.DataFrame({
        "Date": dates,
        "Value": values,
        "Category": np.random.choice(["A", "B", "C"], size=size),
        "Noise": noise
    })
    return df

# Sidebar for data source selection
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV", "Use Sample Data"])

# Initialize dataframe
data = None

# Handle CSV Upload
if data_source == "Upload CSV":
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("File uploaded successfully!")
            
            # Display file info
            st.sidebar.info(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV file to begin.")
else:
    # Use sample data
    data_size = st.sidebar.slider("Sample Size", min_value=10, max_value=1000, value=100)
    data = generate_data(data_size)
    st.sidebar.info("Using generated sample data.")

# Only proceed if we have data
if data is not None:
    # Data preprocessing options
    st.sidebar.header("Data Options")
    
    # Column selection
    if len(data.columns) > 0:
        available_columns = list(data.columns)
        selected_columns = st.sidebar.multiselect("Select columns to use", available_columns, default=available_columns[:4] if len(available_columns) > 4 else available_columns)
        
        if selected_columns:
            data_filtered = data[selected_columns]
        else:
            data_filtered = data
            st.warning("No columns selected. Using all columns.")
    else:
        data_filtered = data
    
    # Handle visualization options
    st.sidebar.header("Visualization")
    
    # Determine numeric columns for plotting
    numeric_columns = data_filtered.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = data_filtered.select_dtypes(exclude=np.number).columns.tolist()
    
    # Display raw data with expander
    with st.expander("View Raw Data"):
        st.dataframe(data_filtered)
        
        # Add download button
        csv = data_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            "filtered_data.csv",
            "text/csv",
            key="download-csv"
        )
    
    # Data Summary
    st.header("Data Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numeric Data Statistics")
        if numeric_columns:
            st.write(data_filtered[numeric_columns].describe())
        else:
            st.info("No numeric columns available for statistics.")
    
    with col2:
        st.subheader("Data Types")
        st.write(pd.DataFrame({
            'Column': data_filtered.columns,
            'Type': data_filtered.dtypes,
            'Non-Null Count': data_filtered.count()
        }))


    st.header("Prompt Optimization")
    col_input, col_output = st.columns(2)
    with col_input:
        # Display dynamic inputs
        st.subheader("Input columns:")

        selected_inputs = st.multiselect(
            'Select input columns',
            selected_columns
        )

    with col_output:
        st.subheader("Output columns:")

        selected_outputs = st.multiselect(
            'Select output columns',
            [c for c in selected_columns if c not in selected_inputs]
        )

    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader("Model Name")
        model_name = st.text_input(f"Your model name", value="openai/gpt-4o-mini")
    
    with col_2:
        st.subheader("Initial Instruction")
        instruction = st.text_input(f"Instruction")
    

    st.subheader("Add input columns:")

    metric_name = st.selectbox(
        'Select metric name',
        ["Exact Match", "Passage Match"]
    )

    metric = None
    if metric_name == "Exact Match":
        metric = dspy.evaluate.metrics.answer_exact_match
    elif metric_name == "Passage Match":
        metric = dspy.evaluate.metrics.answer_passage_match

    import dspy

    lm = dspy.LM(model=model_name, max_tokens=250)

    if metric and selected_inputs and selected_outputs:
        program = dspy.Predict(
            dspy.signatures.make_signature({
                **{
                    input_name: (str, dspy.InputField()) for input_name in selected_inputs
                },
                **{
                    output_name: (str, dspy.OutputField()) for output_name in selected_outputs
                },
            }).with_instructions(instruction),
        )
        # Initialize optimizer
        teleprompter = dspy.teleprompt.MIPROv2(
            metric=metric,
            auto="light",
        )

        fields = list(selected_inputs + selected_outputs)

        trainset = [
            dspy.Example({field: row[field] for field in fields}).with_inputs(*selected_inputs) for _, row in data.iterrows()
        ]

        if st.button("Optimize"):
            with st.spinner("Optimizing..."):
                with dspy.context(lm=lm):
                    optimized_program: dspy.Predict = teleprompter.compile(
                        program,
                        trainset=trainset,
                        max_bootstrapped_demos=2,
                        max_labeled_demos=2,
                        requires_permission_to_run=False,
                    )
            
            states = optimized_program.dump_state()
            st.subheader("Optimized Prompt")
            demos = "\n".join([ 
                f"- {json.dumps({ key: f'`{getattr(demo, key)}`' for key in fields })}" for demo in optimized_program.demos 
            ])
            optimized_prompt = f"""
{optimized_program.signature.instructions}

Your input fields are:
{selected_inputs}

Your output fields are:
{selected_outputs}

This is an example of the task, though some input or output fields are not supplied.

{demos}
"""
            st.write(optimized_prompt)

            st.subheader("Evaluation Score")
            st.write(f"Score: {optimized_program.score}")
else:
    st.write("Please select a data source to begin.")

# Footer
st.markdown("---")
st.caption("Created with Streamlit • Data Explorer Dashboard")