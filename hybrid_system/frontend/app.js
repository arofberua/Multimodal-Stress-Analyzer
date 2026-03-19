document.getElementById('stressForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const loadingDiv = document.getElementById('loading');
    const form = document.getElementById('stressForm');
    const resultDiv = document.getElementById('result');
    
    // Hide form, show loading
    form.style.display = 'none';
    loadingDiv.style.display = 'block';
    
    // Construct payload directly mapped to the explicit ML Feature extraction alias
    const data = {
        sleep_quality: parseInt(document.getElementById('sleep_quality').value),
        headaches: parseInt(document.getElementById('headaches').value),
        academic_performance: parseInt(document.getElementById('academic_performance').value),
        study_load: parseInt(document.getElementById('study_load').value),
        extracurriculars: parseInt(document.getElementById('extracurriculars').value)
    };

    try {
        const response = await fetch('http://127.0.0.1:8000/run_hybrid_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) throw new Error('Failed to fetch from server');
        
        const resData = await response.json();
        
        loadingDiv.style.display = 'none';
        resultDiv.style.display = 'block';
        
        const mlStr = resData.ml_score.toFixed(1);
        const heurStr = resData.heuristic_score.toFixed(1);
        const finalScore = resData.final_score.toFixed(2);
        
        document.getElementById('mlRes').innerText = mlStr;
        document.getElementById('heurRes').innerText = heurStr;
        
        // Populate specific calculation variables showing explicit mathematical fusion step
        document.getElementById('proofMl').innerText = mlStr;
        document.getElementById('proofHeur').innerText = heurStr;
        document.getElementById('proofFinal').innerText = finalScore;
        
        document.getElementById('finalRes').innerText = finalScore;
        
    } catch (error) {
        alert("Error connecting to the backend. Ensure FastAPI is running on port 8000.");
        console.error(error);
        form.style.display = 'block';
        loadingDiv.style.display = 'none';
    }
});
