const API_BASE_URL = "https://models-from-scratch.onrender.com/";

async function testAPI() {
    try {
        let response = await fetch(`${API_BASE_URL}`, {
            method: "GET",
            headers: { "Content-Type": "application/json" },
        });
        let data = await response.json();
        console.log("API Response:", data);

        // Display response on the page
        document.getElementById("output").textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        console.error("Error fetching API:", error);
    }
}

// Attach event listener to button
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("testButton").addEventListener("click", testAPI);
    
    // Automatically fetch API on page load
    testAPI();
});