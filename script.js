async function runInference() {
    const statusElement = document.getElementById('prediction-text');
    statusElement.innerText = "Processing...";

    try {
        // Create an ONNX session
       const modelUrl = './DL_Net_New_model_opset18.onnx';
        const dataUrl = './DL_Net_New_model_opset18.onnx.data';
        // Collect and format inputs based on the CSV structure
        const inputs = [
            parseFloat(document.getElementById('age').value),
            parseFloat(document.getElementById('gender').value),
            parseFloat(document.getElementById('social_media').value),
            parseFloat(document.getElementById('platform').value),
            parseFloat(document.getElementById('sleep').value),
            parseFloat(document.getElementById('screen_time').value),
            parseFloat(document.getElementById('academic').value),
            parseFloat(document.getElementById('physical').value),
            parseFloat(document.getElementById('social_interaction').value),
            parseFloat(document.getElementById('stress').value),
            parseFloat(document.getElementById('anxiety').value),
            parseFloat(document.getElementById('addiction').value)
        ];

        // Prepare the tensor (assuming the model expects shape [1, 12])
        const inputTensor = new ort.Tensor('float32', new Float32Array(inputs), [1, 12]);
        
        // Run the model
        const feeds = { input1: inputTensor }; // 'input1' is the name from the model metadata
        const results = await session.run(feeds);
        
        // Get output (index 0 for non-depressed, index 1 for depressed)
        const output = results[Object.keys(results)[0]].data;
        const prediction = output[0] > 0.5 ? "Positive (Depression Indicated)" : "Negative (No Depression Indicated)";

        statusElement.innerText = prediction;
        statusElement.style.color = output[0] > 0.5 ? "#e74c3c" : "#27ae60";

    } catch (e) {
        console.error(e);
        statusElement.innerText = "Error running model. Check console.";
    }
}
