async function runInference() {
    const statusElement = document.getElementById('prediction-text');
    statusElement.innerText = "Processing...";

    try {
        const modelUrl = './DL_Net_New_model_opset18.onnx';
        const dataUrl = './DL_Net_New_model_opset18.onnx.data';

        const session = await ort.InferenceSession.create(modelUrl, {
            externalData: [{ path: "DL_Net_New_model_opset18.onnx.data", data: dataUrl }]
        });

        // 1. Get Numerical Values
        const age = parseFloat(document.getElementById('age').value);
        const smHours = parseFloat(document.getElementById('social_media').value);
        const sleep = parseFloat(document.getElementById('sleep').value);
        const screen = parseFloat(document.getElementById('screen_time').value);
        const academic = parseFloat(document.getElementById('academic').value);
        const physical = parseFloat(document.getElementById('physical').value);
        const stress = parseFloat(document.getElementById('stress').value);
        const anxiety = parseFloat(document.getElementById('anxiety').value);
        const addiction = parseFloat(document.getElementById('addiction').value);

        // 2. Handle Gender (One-Hot: Male=1,0 or Female=0,1)
        const genderVal = document.getElementById('gender').value;
        const isMale = genderVal === "0" ? 1.0 : 0.0;
        const isFemale = genderVal === "1" ? 1.0 : 0.0;

        // 3. Handle Social Interaction (Assuming the model treats this as a number 0-2)
        const socialInt = parseFloat(document.getElementById('social_interaction').value);

        // 4. Handle Platform (Assuming the last 2 spots are for the platform choice)
        const platformVal = document.getElementById('platform').value;
        let p1 = 0.0, p2 = 0.0;
        if(platformVal === "1") p1 = 1.0; // TikTok
        if(platformVal === "2") p2 = 1.0; // Both

        // 5. Construct the 14-feature array
        // Order: Age, Male, Female, SM_Hours, Sleep, Screen, Acad, Phys, Social_Int, Stress, Anx, Addict, Plat_Extra1, Plat_Extra2
        const inputData = new Float32Array([
            age, isMale, isFemale, smHours, sleep, 
            screen, academic, physical, socialInt, 
            stress, anxiety, addiction, p1, p2
        ]);

        const myInputTensor = new ort.Tensor('float32', inputData, [1, 14]);
        const feeds = { "input1": myInputTensor }; 
        const results = await session.run(feeds);
        
        const output = results[Object.keys(results)[0]].data;
        const score = output[0];
        
        console.log("Model Output Score:", score);

        // A score > 0.5 usually means positive (Depressed)
        // If it's ALWAYS positive, the threshold in your specific model might be different (e.g., 0.0)
        const isDepressed = score > 0.99;
        
        statusElement.innerText = isDepressed ? "Result: Depression Indicated" : "Result: No Depression Indicated";
        statusElement.style.color = isDepressed ? "#e74c3c" : "#27ae60";

    } catch (e) {
        console.error(e);
        statusElement.innerText = "Error: " + e.message;
    }
}
