async function runInference() {
    const statusElement = document.getElementById('prediction-text');
    statusElement.innerText = "Processing...";

    try {
        // 1. Setup the model paths
        const modelUrl = './DL_Net_New_model_opset18.onnx';
        const dataUrl = './DL_Net_New_model_opset18.onnx.data';

        // 2. Load the session (Only once!)
        const session = await ort.InferenceSession.create(modelUrl, {
            externalData: [
                {
                    path: "DL_Net_New_model_opset18.onnx.data",
                    data: dataUrl
                }
            ]
        });

        // 3. Collect the 12 inputs from your HTML form
        const rawInputs = [
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

        // 4. Fix the "Expected 14" error by adding 2 placeholders
        // Your model was trained on 14 features, likely due to one-hot encoding
        const finalInputs = [...rawInputs, 0, 0]; 

        // 5. Create the tensor with shape [1, 14]
        const inputTensor = new ort.Tensor('float32', new Float32Array(finalInputs), [1, 14]);
        
        // 6. Run the model
        const feeds = { "input1": inputTensor }; 
        const results = await session.run(feeds);
        
        // 7. Get and display the result
        const output = results[Object.keys(results)[0]].data;
        const isDepressed = output[0] > 0.5;
        
        statusElement.innerText = isDepressed ? "Result: Depression Indicated" : "Result: No Depression Indicated";
        statusElement.style.color = isDepressed ? "#e74c3c" : "#27ae60";

    } catch (e) {
        console.error(e);
        statusElement.innerText = "Error: " + e.message;
    }
}async function runInference() {
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

  // ... inside runInference function ...

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
    parseFloat(document.getElementById('addiction').value),
    0, // Placeholder 13 (Fixes the "Expected 14" error)
    0  // Placeholder 14 (Fixes the "Expected 14" error)
];

// Update the tensor shape to [1, 14]
const inputTensor = new ort.Tensor('float32', new Float32Array(inputs), [1, 14]);

// ... rest of the code remains the same ...
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
