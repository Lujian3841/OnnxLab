async function runInference() {
    const statusElement = document.getElementById('prediction-text');
    statusElement.innerText = "Locating model files...";

    try {
        // 1. Get the current folder path automatically
        const baseUrl = window.location.href.substring(0, window.location.href.lastIndexOf('/') + 1);
        const modelUrl = baseUrl + 'DL_Net_New_model_opset18.onnx';
        const dataUrl = baseUrl + 'DL_Net_New_model_opset18.onnx.data';

        console.log("Attempting to load model from:", modelUrl);

        // 2. Load the session
        // Note: 'externalData' is required because your model has a separate .data file
        const session = await ort.InferenceSession.create(modelUrl, {
            externalData: [
                {
                    path: "DL_Net_New_model_opset18.onnx.data", // Internal name the model looks for
                    data: dataUrl // External URL where it is actually hosted
                }
            ]
        });

        statusElement.innerText = "Model Loaded! Calculating...";

        // 3. Prepare Inputs (same as before)
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

        const inputTensor = new ort.Tensor('float32', new Float32Array(inputs), [1, 12]);
        
        // IMPORTANT: If 'input1' doesn't work, try just 'input' or 'float_input'
        const feeds = { "input1": inputTensor }; 
        const results = await session.run(feeds);
        
        const output = results[Object.keys(results)[0]].data;
        const isDepressed = output[0] > 0.5;
        
        statusElement.innerText = isDepressed ? "Depression Indicated" : "No Depression Indicated";
        statusElement.style.color = isDepressed ? "#e74c3c" : "#27ae60";

    } catch (e) {
        console.error("Full Error Object:", e);
        statusElement.innerText = "Error: " + e.message;
    }
}
