const API_BASE_URL = "https://models-from-scratch.onrender.com";

// async function testAPI() {
//     try {
//         let response = await fetch(`${API_BASE_URL}/`, {
//             method: "GET",
//             headers: { "Content-Type": "application/json" },
//         });
//         let data = await response.json();
//         console.log("API Response:", data);

//         // Display response on the page
//         document.getElementById("output").textContent = JSON.stringify(data, null, 2);
//     } catch (error) {
//         console.error("Error fetching API:", error);
//     }
// }

// // Attach event listener to button
// document.addEventListener("DOMContentLoaded", () => {
//     document.getElementById("testButton").addEventListener("click", testAPI);
    
//     // Automatically fetch API on page load
//     testAPI();
// });

document.getElementById("fetchButton").addEventListener("click", async () => {
    try {
        // Fetch the API route when the button is clicked
        const response = await fetch(`${API_BASE_URL}/data`);
        if (!response.ok) {
            throw new Error("Network response was not ok");
        }
        const data = await response.json();

        // Display the response data in the page
        document.getElementById("response").innerText = JSON.stringify(data, null, 2);
    } catch (error) {
        console.error("Fetch error:", error);
        document.getElementById("response").innerText = "Error fetching data";
    }
});