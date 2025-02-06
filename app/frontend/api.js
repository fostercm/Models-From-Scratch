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