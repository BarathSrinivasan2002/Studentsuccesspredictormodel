document.getElementById("predictionForm").addEventListener("submit", function (e) {
    e.preventDefault();
  
    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((value, key) => {
      data[key] = value;
    });
  
    sendDataToPredictRoute(data)
      .then((response) => {
        displayResult(response.score);
      })
      .catch((error) => {
        console.error("Error:", error);
        displayError();
      });
  });
  
  function sendDataToPredictRoute(data) {
    // return new Promise((resolve) => {
    //   setTimeout(() => {
    //     const mockResponse = {
    //       score: Math.floor(Math.random() * 101), 
    //       message: "Prediction successful (mock data)",
    //     };
    //     resolve(mockResponse);
    //   }, 1000);
    // });
  
    return fetch('/result', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    }).then((response) => response.json());
    
  }
  
  function displayResult(score) {
    const resultContainer = document.getElementById("result-container");
    const progressCircle = document.getElementById("progressCircle");
    const scoreText = document.getElementById("scoreText");
    const resultText = document.getElementById("resultText");
  
    resultContainer.classList.remove("hidden");
  
    scoreText.textContent = `${score}%`;
    progressCircle.style.background = `conic-gradient(
      ${score > 80 ? '#4caf50' : score >= 50 ? '#ffc107' : '#f44336'} ${score * 3.6}deg,
      #ddd ${score * 3.6}deg
    )`;
  
    if (score > 80) {
      resultText.textContent = "Student Sucess: Excellent!";
      resultText.className = "result-text green";
    } else if (score >= 50) {
      resultText.textContent = "Student Sucess: Good";
      resultText.className = "result-text orange";
    } else {
      resultText.textContent = "Student Sucess: Poor";
      resultText.className = "result-text red";
    }
  }
  function displayError() {
    const resultContainer = document.getElementById("result-container");
    const resultText = document.getElementById("resultText");
  
    resultContainer.classList.remove("hidden");
  
    resultText.textContent = "An error occurred while fetching the prediction.";
    resultText.className = "result-text red";
  }