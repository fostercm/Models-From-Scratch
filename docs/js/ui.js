export function initializeDropdown(selectElement, options) {
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        selectElement.appendChild(optionElement);
    });
}

export function updateUIForTraining(selectedModel, selectedLanguage) {
    document.getElementById('output').textContent = `You selected: ${selectedLanguage} ${selectedModel}`;
    document.getElementById('actionSection').style.display = "block";
}

export function displayTrainSection() {
    document.getElementById('predictSection').style.display = "none";
    document.getElementById('trainSection').style.display = "block";
}

export function displayPredictSection() {
    document.getElementById('trainSection').style.display = "none";
    document.getElementById('predictSection').style.display = "block";
}

export function updateMessage(elementId, message) {
    document.getElementById(elementId).textContent = message;
}