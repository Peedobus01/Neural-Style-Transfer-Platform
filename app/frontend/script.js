document.addEventListener('DOMContentLoaded', () => {
    const contentInput = document.getElementById('content-image');
    const styleInput = document.getElementById('style-image');
    const contentPreview = document.getElementById('content-preview');
    const stylePreview = document.getElementById('style-preview');
    
    // Sliders
    const alphaInput = document.getElementById('alpha');
    const betaInput = document.getElementById('beta');
    const alphaVal = document.getElementById('alpha-val');
    const betaVal = document.getElementById('beta-val');

    alphaInput.addEventListener('input', e => alphaVal.textContent = e.target.value);
    betaInput.addEventListener('input', e => betaVal.textContent = e.target.value);

    // Image previews
    function setupPreview(input, previewElement) {
        input.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewElement.style.backgroundImage = `url('${e.target.result}')`;
                }
                reader.readAsDataURL(e.target.files[0]);
            }
        });
    }

    setupPreview(contentInput, contentPreview);
    setupPreview(styleInput, stylePreview);

    // Stylize Logic
    document.getElementById('stylize-btn').addEventListener('click', async () => {
        if (!contentInput.files[0] || !styleInput.files[0]) {
            alert('Please select both content and style images!');
            return;
        }

        const formData = new FormData();
        formData.append('content_image', contentInput.files[0]);
        formData.append('style_image', styleInput.files[0]);
        formData.append('alpha', alphaInput.value);
        formData.append('beta', betaInput.value);
        formData.append('preserve_colors', document.getElementById('preserve-colors').checked);

        const loader = document.getElementById('loader');
        const resultImage = document.getElementById('result-image');
        
        loader.style.display = 'block';
        resultImage.style.display = 'none';

        try {
            const response = await fetch('http://localhost:8000/api/stylize', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                // Handle the image blob
                const blob = await response.blob();
                const url = URL.createObjectURL(blob);
                resultImage.src = url;
                resultImage.style.display = 'block';
            } else {
                alert('Error processing image.');
            }
        } catch (error) {
            console.error('API Error:', error);
            alert('Failed to connect to the backend.');
        } finally {
            loader.style.display = 'none';
        }
    });
});
