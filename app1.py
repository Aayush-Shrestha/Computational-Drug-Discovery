import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing import image
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, Draw
import pandas as pd
import matplotlib.pyplot as plt
from phar import calculate_rmsd
import os

# Load your QSAR model using joblib
qsar_model = joblib.load('QSAR.h5')
cnn_model = load_model('CNN.h5')

pdb_file1 = 'Zanamivir.pdb'
pdb_file2 = 'Oseltamivir.pdb'
pdb_file3 = 'Peramivir.pdb'
reference_pharmacophores = [pdb_file1, pdb_file2, pdb_file3]

# Function to convert SMILES to PDB
def smiles_to_pdb(smiles):
    output_pdb = 'output.pdb'
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        Chem.MolToPDBFile(mol, output_pdb)
    else:
        print("Invalid SMILES.")

def get_rmsd(smiles):
    smiles_to_pdb(smiles)
    custom_pdb_file = 'output.pdb'
    rmsd_values = calculate_rmsd(custom_pdb_file, reference_pharmacophores)
    rmsd = min(rmsd_values)
    i = rmsd_values.index(rmsd)
    return rmsd,i
   
def smiles_to_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.RDKFingerprint(mol)
    else:
        return None

def generate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    descriptors = [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)]
    return descriptors

def prep_for_QSAR(smiles):
    # Create a DataFrame with the input SMILES
    input_data = pd.DataFrame({'canonical_smiles': [smiles]})
    
    # Perform molecular structure processing
    input_data['fingerprint'] = input_data['canonical_smiles'].apply(smiles_to_mol)
    input_data['descriptors'] = input_data['canonical_smiles'].apply(generate_descriptors)
    input_data[['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']] = pd.DataFrame(input_data['descriptors'].tolist(), index=input_data.index)

    # Generate fingerprint features
    for i in range(2048):
        col_name = f'f{i+1}'
        input_data[col_name] = input_data['fingerprint'].apply(lambda x: x.GetBit(i) if isinstance(x, DataStructs.cDataStructs.ExplicitBitVect) else None).astype(int)

    # Drop unnecessary columns
    input_data = input_data.drop(['canonical_smiles', 'fingerprint', 'descriptors'], axis=1)

    return input_data 

def predict_QSAR(model, preprocessed_data):
    # Make predictions using the loaded model
    predictions = qsar_model.predict(preprocessed_data)

    probabilities = qsar_model.predict_proba(preprocessed_data)[:, 1]
    return predictions, probabilities

# Function to convert SMILES to 2D image
def smiles_to_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)   
    img_path = 'molecule.png'
    img.save(img_path)

def prep_for_CNN(smiles):
    smiles_to_image(smiles)
    image_path = 'molecule.png'
    # Define the target image size
    target_size = (224, 224)
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Make predictions on a new image
def predict_image(smiles):
    
    processed_image = prep_for_CNN(smiles)
    predictions = cnn_model.predict(processed_image)
    return predictions


def main():
    st.set_page_config(
        page_title="Drug Check",
        page_icon="✨",
        layout="wide",  # Set layout to 'wide'
        initial_sidebar_state="expanded",
    )
    # Set Streamlit theme colors
    st.markdown(
        """
        <style>
        .css-1v3fvcr {
            background-color: #ecf0f1;
        }
        .css-1gg0myk {
            background-color: #c5d4e4;
            color: #2c3e5f;
        }
        .css-4lijy7 {
            color: #2c3e5f;
        }
        .css-r57s0o {
            background-color: #7fe0b9;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("DRUG CHECK")

    # Input box for SMILES
    input_smiles = st.text_input("Enter Canonical SMILES:")
    
    # Button to trigger predictions
    if st.button("Predict"):
        # Preprocess for QSAR model
        qsar_input_data = prep_for_QSAR(input_smiles)
        
        # Preprocess for CNN model
        val,i = get_rmsd(input_smiles)
        # Make predictions using QSAR model
        qsar_predictions, qsar_probabilities = predict_QSAR(qsar_model, qsar_input_data)

        # Make predictions using CNN model
        cnn_predictions = predict_image(input_smiles)

        # Display results side by side
        col1, col2 = st.columns(2)

        # Column 1: CNN Model Predictions and Pie Chart
        col1.subheader("CNN Model Predictions:")
        col1.markdown(f"The probability of the compound targeting Neuraminidase is **{round(1 - cnn_predictions[0, 0], 3)}**")
        
        # Plotting pie chart for CNN Model
        # Plotting pie chart for CNN Model
        col1.subheader("CNN Model Probability Distribution:")
        labels_cnn = ['Targets', 'Doesn\'t Target']
        probabilities_cnn = [1 - cnn_predictions[0, 0], cnn_predictions[0, 0]]
        fig_cnn, ax_cnn = plt.subplots()
        ax_cnn.pie(probabilities_cnn, labels=labels_cnn, autopct='%1.1f%%', startangle=90, colors=['#59c0bf', '#66518a'], textprops={'color': '#2c3e50'}, wedgeprops=dict(width=0.3, edgecolor='w'))
        ax_cnn.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        col1.pyplot(fig_cnn, transparent=True)
        
        # Column 2: QSAR Model Predictions and Pie Chart
        col2.subheader("QSAR Model Predictions:")
        col2.markdown(f"The Probability of the compound being active (IC50 <1200 nm) is **{round(1 - qsar_probabilities[0], 3)}**")

        # Plotting pie chart for QSAR Model
        col2.subheader("QSAR Model Probability Distribution:")
        labels_qsar = ['Active', 'Inactive']
        probabilities_qsar = [1 - qsar_probabilities[0], qsar_probabilities[0]]
        fig_qsar, ax_qsar = plt.subplots()
        ax_qsar.pie(probabilities_qsar, labels=labels_qsar, autopct='%1.1f%%', startangle=90, colors=['#59c0bf', '#66518a'], textprops={'color': '#2c3e50'}, wedgeprops=dict(width=0.3, edgecolor='w'))
        ax_qsar.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
        col2.pyplot(fig_qsar, transparent=True)
        
        col3, col4 = st.columns(2)
        # Display molecule image
        col3.subheader("Molecule Image")
        col3.markdown(f"The 2d Structure of the molecule is: ")
        col3.image('molecule.png', caption='Molecule Image', use_column_width=True)
        
        col4.subheader("Pharmacophore RMSD")
        col4.markdown(f"The lowest RMSD is **{round(val, 3)}** similar to {os.path.basename(reference_pharmacophores[i])}")
        col4.image(f'png{i+1}.png', caption=os.path.basename(reference_pharmacophores[i]), use_column_width=True)

        
        
if __name__ == "__main__":
    main()