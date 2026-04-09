async function runInference() {
    const statusElement = document.getElementById('prediction-text');
    statusElement.innerText = "Processing...";

    try {
        // 1. Setup paths
        const modelUrl = './DL_Net_New_model_opset18.onnx';
        const dataUrl = './DL_Net_New_model_opset18.onnx.data';

        // 2. Load the session (Only one session allowed)
        const session = await ort.InferenceSession.create(modelUrl, {
            externalData: [
                {
                    path: "DL_Net_New_model_opset18.onnx.data",
                    data: dataUrl
                }
            ]
        });

        // 3. Get values from your HTML inputs
        const age = parseFloat(document.getElementById('age').value);
        const gender = parseFloat(document.getElementById('gender').value);
        const socialMedia = parseFloat(document.getElementById('social_media').value);
        const platform = parseFloat(document.getElementById('platform').value);
        const sleep = parseFloat(document.getElementById('sleep').value);
        const screenTime = parseFloat(document.getElementById('screen_time').value);
        const academic = parseFloat(document.getElementById('academic').value);
        const physical = parseFloat(document.getElementById('physical').value);
        const socialInt = parseFloat(document.getElementById('social_interaction').value);
        const stress = parseFloat(document.getElementById('stress').value);
        const anxiety = parseFloat(document.getElementById('anxiety').value);
        const addiction = parseFloat(document.getElementById('addiction').value);

        // 4. Create the 14-value array the model expects
        // We take your 12 inputs and add 2 placeholders (0, 0)
        const inputData = new Float32Array([
            age, gender, socialMedia, platform, sleep, 
            screenTime, academic, physical, socialInt, 
            stress, anxiety, addiction, 0, 0
        ]);

        // 5. Create the single Tensor [1, 14]
        const finalTensor = new ort.Tensor('float32', inputData, [1, 14]);
        
        // 6. Run the model using the input name 'input1'
        const feeds = { "input1": finalTensor }; 
        const results = await session.run(feeds);
        
        // 7. Process result
        const output = results[Object.keys(results)[0]].data;
        const isDepressed = output[0] > 0.5;
        
        statusElement.innerText = isDepressed ? "Result: Depression Indicated" : "Result: No Depression Indicated";
        statusElement.style.color = isDepressed ? "#e74c3c" : "#27ae60";

    } catch (e) {
        console.error("Full error details:", e);
        statusElement.innerText = "Error: " + e.message;
    }
}
