// API Base URL
const API_BASE_URL = "http://127.0.0.1:8000"

// List of models and languages
const models = [
    "Linear Regression",
];

const languages = [
    "Python",
    "C",
];

// Get the dropdown elements
const modelSelect = document.getElementById('modelSelect');
const languageSelect = document.getElementById('languageSelect');
const actionSection = document.getElementById('actionSection');
const trainSection = document.getElementById('trainSection');
const predictSection = document.getElementById('predictSection');

// Populate the model dropdown
models.forEach(model => {
    const option = document.createElement('option');
    option.value = model;
    option.textContent = model;
    modelSelect.appendChild(option);
});

// Populate the language dropdown
languages.forEach(language => {
    const option = document.createElement('option');
    option.value = language;
    option.textContent = language;
    languageSelect.appendChild(option);
});

// Select Model
document.getElementById('selectBtn').addEventListener('click', () => {
    const selectedModel = modelSelect.value;
    const selectedLanguage = languageSelect.value;
    document.getElementById('output').textContent = `You selected: ${selectedLanguage} ${selectedModel}`;
    actionSection.style.display = "block";
});

// Train
document.getElementById('trainModel').addEventListener('click', () => {
    predictSection.style.display = "none";
    trainSection.style.display = "block";
});

document.getElementById('trainBtn').addEventListener('click', () => {
    // Validate data
    const indicatorData = document.getElementById("indicatorData").files[0];
    const responseData = document.getElementById("responseData").files[0];

    // Check if both files are uploaded
    if (!indicatorData || !responseData) {
        alert("Please upload both indicator and response data files.");
        return;
    }

    // Check if the files are CSV
    if (!indicatorData.name.toLowerCase().endsWith('.csv') || !responseData.name.toLowerCase().endsWith('.csv')) {
        alert("Please upload CSV files only.");
        return;
    }

    // Give training message
    const selectedModel = modelSelect.value;
    const selectedLanguage = languageSelect.value;
    document.getElementById('trainMessage1').textContent = `Training ${selectedLanguage} ${selectedModel} Model...`;

    // Store the data for API call
    const formData = new FormData();
    formData.append('model_name', selectedModel);
    formData.append('language', selectedLanguage);
    formData.append('indicator', indicatorData);
    formData.append('response', responseData);

    // Call the trainModel function and download the trained model
    trainModel(formData).then(result => {
        downloadJSON(result);
    });

    // Notify user
    document.getElementById('trainMessage2').textContent = `Model trained successfully!`;
});

// Predict
document.getElementById('predictModel').addEventListener('click', () => {
    trainSection.style.display = "none";
    predictSection.style.display = "block";
});

document.getElementById('predictBtn').addEventListener('click', () => {
    // Validate data
    const modelFile = document.getElementById("modelFile").files[0];
    const inputData = document.getElementById("inputData").files[0];

    // Check if both files are uploaded
    if (!indicatorData || !responseData) {
        alert("Please upload both model and input data files.");
        return;
    }

    // Check if the files are the correct formats
    if (!modelFile.name.toLowerCase().endsWith('.json')) {
        alert("Please upload a JSON model file.");
        return;
    }

    if (!inputData.name.toLowerCase().endsWith('.csv')) {
        alert("Please upload a CSV input data file.");
        return;
    }

    // Give prediction message
    const selectedModel = modelSelect.value;
    const selectedLanguage = languageSelect.value;
    document.getElementById('predictMessage1').textContent = `Predicting with ${selectedLanguage} ${selectedModel} Model...`;

    // Store the data for API call
    const formData = new FormData();
    formData.append('model_name', selectedModel);
    formData.append('language', selectedLanguage);
    formData.append('indicator', inputData);
    formData.append('model_file', modelFile);

    // Call the predictModel function and display the results
    predictModel(formData).then(result => {
        downloadCSV(result);
        console.log(result);
        document.getElementById('predictMessage2').textContent = `Predictions saved to predictions.csv`;
    });
    
});

// Train Model
async function trainModel(formData) {
    // Fetch API
    const response = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        body: formData,
    });

    // Handle response
    const result = await response.json();

    return result;

}

// Predict Model
async function predictModel(formData) {
    // Fetch API
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
    });

    // Handle response
    const result = await response.json();

    return result;
}

// Save JSON
function downloadJSON(data) {

    // Convert JSON object to string
    const jsonString = JSON.stringify(data, null, 4);

    // Create a Blob (Binary Large Object) from the JSON string
    const blob = new Blob([jsonString], { type: "application/json" });

    // Create a temporary link element
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "model.json";

    // Trigger the download
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Save CSV
function downloadCSV(data) {

    // Convert JSON object to CSV string
    const csvString = Papa.unparse(data.predictions);

    // Create a Blob (Binary Large Object) from the CSV string
    const blob = new Blob([csvString], { type: "text/csv" });

    // Create a temporary link element
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "predictions.csv";

    // Trigger the download
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}