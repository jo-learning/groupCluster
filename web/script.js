document.getElementById("playerForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const desiredServices = JSON.parse(document.getElementById("desiredServices").value);
  const level = document.getElementById("level").value;
  const rank = parseFloat(document.getElementById("rank").value);
  const maxBudgetPerSession = parseFloat(document.getElementById("budget").value);
  const travelDistance = parseFloat(document.getElementById("travel").value);
  const goals = JSON.parse(document.getElementById("goals").value);
  const languages = JSON.parse(document.getElementById("languages").value);

  const playerData = {
    desiredServices,
    level,
    rank,
    maxBudgetPerSession,
    travelDistance,
    goals,
    languages
  };

  const res = await fetch("http://127.0.0.1:5000/cluster-player", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(playerData)
  });

  const data = await res.json();
  if (data.error) {
    document.getElementById("result").innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
  } else {
    const recommendations = data.recommendedPlayers
      .map(p => `<li>${p.id} - Level: ${p.level}, Rank: ${p.rank}, Budget: $${p.maxBudgetPerSession}</li>`)
      .join("");

    document.getElementById("result").innerHTML = `
      <p><strong>Predicted Cluster:</strong> ${data.cluster}</p>
      <p><strong>Similar Players:</strong></p>
      <ul>${recommendations}</ul>
    `;
  }
});
