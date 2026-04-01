import streamlit as st
import pandas as pd
from transformers import pipeline
import io
import openpyxl

# Load pre-trained model
@st.cache_resource
def load_model():
    model = pipeline("text-classification", model="Group-7/Assignment_1")
    return model

# Load uploaded file
def load_data(file):
    if file.name.endswith(".xlsx"):
        df = pd.read_excel(file, engine="openpyxl")
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        st.error("Unsupported file format. Please upload .xlsx or .csv file.")
        return None
    return df

# Main app
def main():
    st.title("News Classification App")

    uploaded_file = st.file_uploader(
        "Upload an Excel or CSV file",
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is None:
            return

        if "content" not in df.columns:
            st.error("No 'content' column found in the uploaded file. Please ensure the file contains a column named 'content'.")
            return

        # Rename column nicely
        df.rename(columns={"content": "News Content"}, inplace=True)

        st.write("### Uploaded Data")
        st.dataframe(df.head())

        # Convert content column to clean list of strings
        text_to_classify = df["News Content"].fillna("").astype(str).tolist()

        # Load model
        model = load_model()

        # Get predictions
        predictions = model(text_to_classify)

        # Optional raw output
        show_raw = st.checkbox("Show raw prediction output")
        if show_raw:
            st.write("### Raw Predictions Output")
            st.write(predictions)

        # Map labels to real class names
        label_map = {
            "LABEL_0": "Business",
            "LABEL_1": "Opinion",
            "LABEL_2": "Political gossip",
            "LABEL_3": "Sports",
            "LABEL_4": "World news"
        }

        predicted_labels = [
            label_map.get(pred["label"], pred["label"]) for pred in predictions
        ]

        confidence_scores = [
            round(pred["score"] * 100, 2) for pred in predictions
        ]

        st.write("### Predictions")
        st.dataframe(pd.DataFrame({
            "Predicted Class": predicted_labels,
            "Confidence (%)": confidence_scores
        }))

        # Add predictions back to dataframe
        df["Predicted Class"] = predicted_labels
        df["Confidence (%)"] = confidence_scores

        # Class distribution chart
        st.write("### Class Distribution")
        class_counts = pd.Series(predicted_labels).value_counts()
        st.bar_chart(class_counts)

        st.write("### Classified Data")
        st.dataframe(df)

        # Download classified data
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Classified Data",
            data=io.BytesIO(csv_data),
            file_name="classified_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()