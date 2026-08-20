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
        formData.append('num_steps', document.getElementById('num-steps').value);
        formData.append('intermediate_frames', document.getElementById('intermediate-frames').value);
        formData.append('preserve_colors', document.getElementById('preserve-colors').checked);

        const loader = document.getElementById('loader');
        const resultImage = document.getElementById('result-image');
        const progressContainer = document.getElementById('progress-container');
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const stylizeBtn = document.getElementById('stylize-btn');
        
        loader.style.display = 'block';
        resultImage.style.display = 'none';
        progressContainer.style.display = 'block';
        progressFill.style.width = '0%';
        progressText.textContent = 'Initializing Model...';
        stylizeBtn.disabled = true;

        try {
            const response = await fetch('/api/stylize', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                alert('Error starting style transfer.');
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                let lines = buffer.split("\n\n");
                
                // Keep the last incomplete chunk in the buffer
                buffer = lines.pop(); 

                for (let line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.substring(6); // Remove "data: "
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.error) {
                                alert("Error: " + data.error);
                                break;
                            }
                            
                            // Update progress
                            if (data.step && data.total) {
                                loader.style.display = 'none';
                                const percent = (data.step / data.total) * 100;
                                progressFill.style.width = `${percent}%`;
                                progressText.textContent = `Epoch ${data.step} / ${data.total}`;
                            }
                            
                            // Update image if present
                            if (data.image) {
                                resultImage.style.opacity = '0'; // For smooth crossfade
                                setTimeout(() => {
                                    resultImage.src = data.image;
                                    resultImage.style.display = 'block';
                                    resultImage.style.opacity = '1';
                                }, 50);
                            }
                            
                        } catch (e) {
                            console.error("Failed to parse SSE event:", e, dataStr);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('API Error:', error);
            alert('Failed to connect to the backend.');
        } finally {
            loader.style.display = 'none';
            stylizeBtn.disabled = false;
        }
    });
});
