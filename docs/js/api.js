import {API_BASE_URL} from './constants.js';

export async function callApi(url, formData) {
    const response = await fetch(url, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error('API request failed: ' + response.statusText);
    }

    return response.json();
}

export async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        const statusBox = document.getElementById("api-status");

        if (data.status === "up") {
            statusBox.className = "status-box status-up";
            statusBox.textContent = "API is UP";
        } else {
            statusBox.className = "status-box status-down";
            statusBox.textContent = "API is DOWN";
        }
    } catch (error) {
        const statusBox = document.getElementById("api-status");
        statusBox.className = "status-box status-down";
        statusBox.textContent = "API is DOWN";
    }
}