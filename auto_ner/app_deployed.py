'''
Creator: Sudhir Arvind Deshmukh
Run command: streamlit run app.py 
'''
import streamlit as st
import spacy
from spacy.tokens import Doc
from spacy.training.example import Example
import datetime
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
from transformers import AutoTokenizer, T5ForConditionalGeneration
from spacy import displacy
import subprocess

## Load spaCy models from saved_models directory

# Get absolute path to the current script's directory

def ensure_spacy_models_installed():
    models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    for model in models:
        try:
            # Try loading the model, this will raise an exception if it's not installed
            spacy.load(model)
        except OSError:
            print(f"Installing {model}...")
            subprocess.call(["python", "-m", "spacy", "download", model])


def ensure_folders_exist(script_dir):
    images_path = os.path.join(script_dir, "images")
    saved_model_path = os.path.join(script_dir, "saved_models")

    # Create the 'images' directory if it doesn't exist
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Create the 'saved_model' directory if it doesn't exist
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)

ensure_spacy_models_installed()

# Get absolute path to the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ensure that required folders exist
ensure_folders_exist(script_dir)
saved_models_dir = os.path.join(script_dir, "saved_models")
nlp_models = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"] + [os.path.join(saved_models_dir, f"{model_name}") for model_name in os.listdir(saved_models_dir)]

# fuction to load the csv file and extract sentences and tags
def load_data_from_csv(file):
    df = pd.read_csv(file, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")
    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    tags = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, tags



# Streamlit UI for Online Inference
def online_inference():
    st.title("Online Inference")
    
    selected_model = st.selectbox("Select base Model for finetunning", nlp_models)

    # Load the selected spaCy model
    # model_path = os.path.join(saved_models_dir, f"{selected_model}")
    nlp = spacy.load(selected_model)


    text_input = st.text_input("Enter Text for Inference")

    if text_input:
        doc = nlp(text_input)
        
        # Filter out 'O' entities and get unique entity types
        filtered_entities = [ent for ent in doc.ents if ent.label_ != 'O']
        unique_entity_types = list(set(ent.label_ for ent in filtered_entities))
        
        if filtered_entities:
            # Define Google-themed colors for each entity type
            color_dict = {
                'B-geo': '#4285F4',  # Blue
                'B-gpe': '#EA4335',  # Red
                'B-per': '#FBBC05',  # Yellow
                'I-geo': '#0F9D58',  # Green
                'B-org': '#34A853',  # Green
                'I-org': '#FF9800',  # Orange
                'B-tim': '#AA66CC',  # Purple
                'B-art': '#FFC107',  # Amber
                'I-art': '#9C27B0',  # Purple
                'I-per': '#03A9F4',  # Blue
                'I-gpe': '#009688',  # Teal
                'I-tim': '#FF5722',  # Deep Orange
                'B-nat': '#7B1FA2',  # Deep Purple
                'B-eve': '#8BC34A',  # Light Green
                'I-eve': '#FDD835',  # Yellow
                'I-nat': '#616161'   # Gray
            }
            
            # Render the visualization with custom colors
            options = {"ents": unique_entity_types, "colors": color_dict}
            html = spacy.displacy.render(doc, style="ent", options=options)
            st.components.v1.html(html, height=400)
        else:
            st.write("No named entities found in the text.")
    
# Streamlit UI for Model Training
def model_training():
    
    st.title("Model Training")
    
    base_model = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
    selected_model = st.selectbox("Select base Model to Train", base_model)
    
    # Define hyperparameters
    learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.1, step=0.001, value=0.01)
    n_iter = st.slider("Number of Iterations", min_value=1, max_value=10, value=2)
    dropout = st.slider("Dropout", min_value=0.1, max_value=0.9, step=0.1, value=0.5)
    
    uploaded_file = st.file_uploader("Upload Training Data (CSV)", type="csv")
    
    model_name_uniq = st.text_input("Enter Model Name")
    if st.button("Train & Evaluate Model"):
        if uploaded_file is not None:
            
            # Load training data from the uploaded CSV file
            sentences, tags = load_data_from_csv(uploaded_file)
            
            # Split data into training, validation, and test sets
            train_sentences, test_sentences, train_tags, test_tags = train_test_split(sentences, tags, test_size=0.2, random_state=42)
            train_sentences, val_sentences, train_tags, val_tags = train_test_split(train_sentences, train_tags, test_size=0.2, random_state=42)

            print(f"Experimenting with model: {selected_model}")


            # Load the pre-trained model
            nlp = spacy.load(selected_model)

            # Add or modify the NER component in the pipeline
            if "ner" not in nlp.pipe_names:
                ner = nlp.add_pipe("ner")
            else:
                ner = nlp.get_pipe("ner")

             # Function to convert input format to spaCy format
            def convert_to_spacy_format(sentences, tags):
                examples = []
                for sent, tag_list in zip(sentences, tags):
                    words = sent
                    spaces = [True] * len(words)
                    doc = Doc(nlp.vocab, words=words, spaces=spaces)
                    gold_entities = []
                    for token, tag in zip(doc, tag_list):
                        start = token.idx
                        end = start + len(token.text)
                        gold_entities.append((start, end, tag))
                    example = Example.from_dict(doc, {"entities": gold_entities})
                    examples.append(example)
                return examples

            # Add entity labels to the ner component
            for label in set(tag for tag_list in tags for tag in tag_list):
                ner.add_label(label)

            # Create spaCy examples for training
            train_examples = convert_to_spacy_format(train_sentences, train_tags)
            val_examples = convert_to_spacy_format(val_sentences, val_tags)


            # Lists to store learning curve data
            train_losses = []
            val_precisions = []
            val_recalls = []

            total_batches = len(train_examples) / 8
            ner_metrics = []
            # Train the NER model
            for epoch in range(n_iter):
                random.shuffle(train_examples)
                st.write("this is iteration number:", epoch)
                losses = {}
                progress_bar = st.progress(0) 
                for batch_index, batch in enumerate(spacy.util.minibatch(train_examples, size=8), start=1):
                    nlp.update(batch, drop=dropout, losses=losses)
                    # Calculate progress percentage
                    progress_percentage = batch_index / (total_batches + 1)
                    progress_bar.progress(progress_percentage)  # Display progress in Streamlit
                train_losses.append(losses["ner"])

                # Evaluate the model on the validation set
                metrics = nlp.evaluate(val_examples)
                val_precisions.append(metrics["ents_p"])
                val_recalls.append(metrics["ents_r"])

            # Append metrics to the ner_metrics list
            ner_metrics.append(metrics)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_model_name = f"{model_name_uniq}_ner_model_{current_time}"
            # Plot learning curve
            plt.figure(figsize=(12, 4))
            plt.plot(range(n_iter), train_losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"Learning Curve for Model: {save_model_name}")
            plt.legend()
            learning_curve_plot_path = f"learning_curve_{save_model_name}.png"
            plt.savefig(learning_curve_plot_path)
            st.image(learning_curve_plot_path)

            # Plot Precision-Recall curve
            plt.figure(figsize=(12, 4))
            plt.plot(val_recalls, val_precisions, label="Precision-Recall Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve for Model: {save_model_name}")
            plt.legend()
            pr_curve_plot_path = f"precision_recall_curve_{save_model_name}.png"
            plt.savefig(pr_curve_plot_path)
            st.image(pr_curve_plot_path)

            # Save the trained model to disk with timestamp

            nlp.to_disk(saved_models_dir + save_model_name)
            st.success(f"Trained model saved as: {save_model_name}")

            # Print important NER performance metrics
            ner_performance_metrics = ["ents_p", "ents_r", "ents_f", 
                                       #"ents_per_type"
                                       ]
            # Print model performance metrics
            st.write("---") 
            st.subheader("Evaluation Metrics on validation data")
            for model_name, metrics in zip([selected_model], ner_metrics):
                st.write(f"Model: {model_name}")
                for metric_name in ner_performance_metrics:
                    metric_value = metrics.get(metric_name, 0.0)
                    st.write(f"{metric_name}: {metric_value}")
                st.write("")  # Add an empty line for spacing
            st.write("---") 
            st.subheader("Performance Metrics on test data")
 
            test_examples = convert_to_spacy_format(test_sentences, test_tags)
            # Evaluate the model on the validation set

            test_metrics = nlp.evaluate(test_examples)
            # Print important NER performance metrics
            # ner_performance_metrics = ["ents_p", "ents_r", "ents_f"]
            # Print model performance metrics
                
            for metric_name in ner_performance_metrics:
                metric_value = test_metrics.get(metric_name, 0.0)
                st.write(f"{metric_name}: {metric_value}")

            st.write("---")

        else:
            st.warning("Please upload training data in CSV format.")

def gen_ai():


    # Streamlit app layout
    st.title("Few-Shot Named Entity Recognition with Flan")

    # Load the Flan model
    model_name = st.selectbox("Select Flan Model", ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"])
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load a pre-trained tokenizer that's compatible with T5
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    st.write("---") 
    # User input for few-shot examples
    st.subheader("Few-Shot Examples")
    examples = []
    num_examples = st.number_input("Number of Examples", min_value=1, value=2)
    for _ in range(num_examples):
        col1, col2 = st.columns([3, 1]) 
        with col1:
            example_text = st.text_input(f"Example {_+1} (Text)")
        with col2:
            example_label = st.text_input(f"Example {_+1} (Label)")
        if example_text and example_label:
            examples.append((example_text, example_label))
    st.write("---") 
    # User input for query text
    st.subheader("Query Text")
    query = st.text_input("Enter Query Text")

    # Detect Entities button
    detect_button = st.button("Detect Entities")

    # Generate named entities
    if detect_button:
        if not examples or not query:
            st.warning('Need both examples and query as user input', icon="⚠️")
        prompt = "\n".join([f"NER: {example[0]} Labels: {example[1]}" for example in examples])
        prompt += f"\n{query} Labels:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Process the generated output for displacy
        entities = generated_text.split("Labels:")
        entities = [e.strip().split(":")[0].strip() for e in entities if e.strip()]
        st.write("---") 
        # Display identified named entities
        st.subheader("Identified Named Entities:")
        
        doc = {"text": query, "ents": [{"start": query.find(entity), "end": query.find(entity) + len(entity), "label": "Custom Entity"} for entity in entities], "title": None}
        html = displacy.render(doc, style="ent", manual=True, minify=True)
        st.components.v1.html(html)
        st.write("---") 
        st.write(doc)

def main():

    # Streamlit App
    st.set_page_config(page_title="NER Model Experimentation")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Online Inference", "Model Training", 
                                    #"Evaluation Metrics", 
                                    "GEN AI"])

    if page == "Online Inference":
        online_inference()
    elif page == "Model Training":
        model_training()
    elif page == "GEN AI":
        gen_ai()


# call main fuction
if __name__=="__main__":
    main() 

