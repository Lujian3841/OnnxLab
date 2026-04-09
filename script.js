async function runInference() {
    const statusElement = document.getElementById('prediction-text');
    statusElement.innerText = "Processing...";

    try {
        // 1. Setup paths - Using the exact names from your GitHub
        const modelUrl = './DL_Net_New_model_opset18.onnx';
        const dataUrl = './DL_Net_New_model_opset18.onnx.data';

        // 2. Load the session with external weights
        const session = await ort.InferenceSession.create(modelUrl, {
            externalData: [
                {
                    path: "DL_Net_New_model_opset18.onnx.data",
                    data: dataUrl
                }
            ]
        });

        // 3. Gather the 12 inputs from your HTML
        const age = parseFloat(document.getElementById('age').value) || 0;
        const gender = parseFloat(document.getElementById('gender').value) || 0;
        const socialMedia = parseFloat(document.getElementById('social_media').value) || 0;
        const platform = parseFloat(document.getElementById('platform').value) || 0;
        const sleep = parseFloat(document.getElementById('sleep').value) || 0;
        const screenTime = parseFloat(document.getElementById('screen_time').value) || 0;
        const academic = parseFloat(document.getElementById('academic').value) || 0;
        const physical = parseFloat(document.getElementById('physical').value) || 0;
        const socialInt = parseFloat(document.getElementById('social_interaction').value) || 0;
        const stress = parseFloat(document.getElementById('stress').value) || 0;
        const anxiety = parseFloat(document.getElementById('anxiety').value) || 0;
        const addiction = parseFloat(document.getElementById('addiction').value) || 0;

        // 4. Build the 14-feature array the model requires
        // We add two '0' placeholders at the end to satisfy the model's 14-input requirement
        const inputData = new Float32Array([
            age, gender, socialMedia, platform, sleep, 
            screenTime, academic, physical, socialInt, 
            stress, anxiety, addiction, 0, 0
        ]);

        // 5. Create the tensor [1, 14] - named uniquely to avoid redeclaration errors
        const myInputTensor = new ort.Tensor('float32', inputData, [1, 14]);
        
        // 6. Run the model using the internal name 'input1'
        const feeds = { "input1": myInputTensor }; 
        const results = await session.run(feeds);
        
        // 7. Extract the result
        const output = results[Object.keys(results)[0]].data;
        const depressionScore = output[0]; // Assuming index 0 is the depression probability
        
        const isDepressed = depressionScore > 0.5;
        
        statusElement.innerText = isDepressed ? "Result: Depression Indicated" : "Result: No Depression Indicated";
        statusElement.style.color = isDepressed ? "#e74c3c" : "#27ae60";

    } catch (e) {
        console.error("Inference Error:", e);
        statusElement.innerText = "Error: " + e.message;
    }
}
