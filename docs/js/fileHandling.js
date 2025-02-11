export const VALID_FILE_TYPES = {
    CSV: ".csv",
    JSON: ".json",
};

// Validate file inputs
export function validateFileInput(file, expectedExtension) {
    if (!file) {
        throw new Error(`Please upload a valid ${expectedExtension} file.`);
    }
    if (!file.name.toLowerCase().endsWith(expectedExtension)) {
        throw new Error(`Please upload a ${expectedExtension} file.`);
    }
}

// Download JSON file
export function downloadJSON(data) {
    const jsonString = JSON.stringify(data, null, 4);
    const blob = new Blob([jsonString], { type: "application/json" });
    triggerDownload(blob, "model.json");
}

// Download CSV file
export function downloadCSV(data) {
    const csvString = Papa.unparse(data.predictions);
    const blob = new Blob([csvString], { type: "text/csv" });
    triggerDownload(blob, "predictions.csv");
}

// Trigger file download
function triggerDownload(blob, filename) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}