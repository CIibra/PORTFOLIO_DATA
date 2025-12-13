import pandas as pd

# Charger données
df = pd.read_csv("suivis.csv")

# Calcul KPI
total = len(df)
completes = (df["statut"] == "complété").sum()
taux_completion = completes / total if total else 0
presence_moy = df["presence_pct"].mean()

kpi = {
    "Total suivis": total,
    "Taux de complétion": f"{taux_completion:.0%}",
    "Présence moyenne": f"{presence_moy:.0%}"
}

print("✅ KPI calculés:", kpi)

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Export Excel
pd.DataFrame([kpi]).to_excel("rapport.xlsx", index=False)

# Export PDF
c = canvas.Canvas("rapport.pdf", pagesize=A4)
c.setFont("Helvetica", 14)
c.drawString(100, 750, "Rapport Employabilité")
y = 700
for key, value in kpi.items():
    c.drawString(100, y, f"{key}: {value}")
    y -= 30
c.save()

print("✅ Rapport Excel et PDF générés.")