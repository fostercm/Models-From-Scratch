import { API_BASE_URL, VALID_FILE_TYPES, MODELS, LANGUAGES } from './constants.js';
import { callApi, checkAPIStatus} from './api.js';
import { validateFileInput, downloadJSON, downloadCSV } from './fileHandling.js';
import { initializeDropdown, updateUIForTraining, displayTrainSection, displayPredictSection, updateMessage } from './ui.js';

// DOM Elements
const modelSelect = document.getElementById('modelSelect');
const languageSelect = document.getElementById('languageSelect');
const trainSection = document.getElementById('trainSection');
const predictSection = document.getElementById('predictSection');

// Initialize dropdowns
initializeDropdown(modelSelect, MODELS);
initializeDropdown(languageSelect, LANGUAGES);

// Event listeners
document.getElementById('selectBtn').addEventListener('click', handleModelLanguageSelection);
document.getElementById('trainModel').addEventListener('click', displayTrainSection);
document.getElementById('predictModel').addEventListener('click', displayPredictSection);
document.getElementById('trainBtn').addEventListener('click', handleTrainModel);
document.getElementById('predictBtn').addEventListener('click', handlePredictModel);

// Handle model and language selection
function handleModelLanguageSelection() {
    const selectedModel = modelSelect.value;
    const selectedLanguage = languageSelect.value;
    updateUIForTraining(selectedModel, selectedLanguage);
}

// Handle training model
async function handleTrainModel() {
    try {
        const indicatorData = document.getElementById("indicatorData").files[0];
        const responseData = document.getElementById("responseData").files[0];

        validateFileInput(indicatorData, VALID_FILE_TYPES.CSV);
        validateFileInput(responseData, VALID_FILE_TYPES.CSV);

        const selectedModel = modelSelect.value;
        const selectedLanguage = languageSelect.value;

        updateMessage('trainMessage1', `Training ${selectedLanguage} ${selectedModel} Model...`);

        const formData = new FormData();
        formData.append('model_name', selectedModel);
        formData.append('language', selectedLanguage);
        formData.append('indicator', indicatorData);
        formData.append('response', responseData);

        const result = await callApi(`${API_BASE_URL}/train`, formData);
        downloadJSON(result);

        updateMessage('trainMessage2', `Model trained successfully!`);
    } catch (error) {
        alert(error.message);
    }
}

// Handle prediction model
async function handlePredictModel() {
    try {
        const modelFile = document.getElementById("modelFile").files[0];
        const inputData = document.getElementById("inputData").files[0];

        validateFileInput(modelFile, VALID_FILE_TYPES.JSON);
        validateFileInput(inputData, VALID_FILE_TYPES.CSV);

        const selectedModel = modelSelect.value;
        const selectedLanguage = languageSelect.value;

        updateMessage('predictMessage1', `Predicting with ${selectedLanguage} ${selectedModel} Model...`);

        const formData = new FormData();
        formData.append('model_name', selectedModel);
        formData.append('language', selectedLanguage);
        formData.append('indicator', inputData);
        formData.append('model_file', modelFile);

        const result = await callApi(`${API_BASE_URL}/predict`, formData);
        downloadCSV(result);

        updateMessage('predictMessage2', `Predictions saved to predictions.csv`);
    } catch (error) {
        alert(error.message);
    }
}

// Check API status
document.addEventListener("DOMContentLoaded", () => {
    checkAPIStatus();
});