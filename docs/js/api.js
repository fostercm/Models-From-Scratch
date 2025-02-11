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
    const statusBox = document.getElementById("api-status");

    const timeout = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Request timed out")), 5000)
    );

    try {
        console.log("Checking API status...");
        const response = await Promise.race([
            fetch(`${API_BASE_URL}/health`),
            timeout
        ]);
        const data = await response.json();


        if (data.status === "up") {
            console.log("API is UP");
            statusBox.className = "status-box status-up";
            statusBox.textContent = "API is UP";
        } else {
            console.log("API is DOWN");
            statusBox.className = "status-box status-down";
            statusBox.textContent = "API is DOWN";
        }
    } catch (error) {
        console.error("Error checking API status:", error);
        statusBox.className = "status-box status-down";
        statusBox.textContent = "API is DOWN";
    }
}