import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# 1. Styling for a pro look
# ----------------------------
sns.set_style("whitegrid")          # clean background
sns.set_context("talk", font_scale=1.0)  # good for presentations

# ----------------------------
# 2. Generate realistic data
# ----------------------------

rng = np.random.default_rng(42)
n_customers = 200

channels = rng.choice(
    ["Paid Search", "Social", "Email", "Affiliate"],
    size=n_customers,
    p=[0.35, 0.30, 0.20, 0.15]
)

# Typical acquisition cost by channel (in dollars)
channel_cac_means = {
    "Paid Search": 120,   # expensive but targeted
    "Social": 80,
    "Email": 40,          # cheap
    "Affiliate": 60,
}

# Generate CAC with some noise
acquisition_cost = np.array([
    rng.normal(channel_cac_means[ch], 15) for ch in channels
])
acquisition_cost = np.clip(acquisition_cost, 10, None)

# Lifetime value roughly 5–7x CAC depending on channel, plus noise
channel_multipliers = {
    "Paid Search": 6.5,
    "Social": 5.5,
    "Email": 7.0,
    "Affiliate": 6.0,
}

customer_lifetime_value = np.array([
    acquisition_cost[i] * channel_multipliers[channels[i]] + rng.normal(0, 60)
    for i in range(n_customers)
])
customer_lifetime_value = np.clip(customer_lifetime_value, 50, None)

# Build DataFrame
df = pd.DataFrame({
    "Acquisition_Cost": acquisition_cost,
    "Customer_Lifetime_Value": customer_lifetime_value,
    "Channel": channels,
})

# ----------------------------
# 3. Create Seaborn scatterplot
# ----------------------------

# 8x8 inches with dpi=64 → 512x512 pixels
plt.figure(figsize=(8, 8))

ax = sns.scatterplot(
    data=df,
    x="Acquisition_Cost",
    y="Customer_Lifetime_Value",
    hue="Channel",
    style="Channel",
    s=70,
    edgecolor="white",
    alpha=0.9,
)

# Add a reference line: CLV = 6 * CAC (good rule-of-thumb line)
x_min, x_max = ax.get_xlim()
x_line = np.linspace(x_min, x_max, 100)
ax.plot(x_line, 6 * x_line, "--", linewidth=1, color="gray", label="6x CLV guideline")

# Titles and labels
ax.set_title("Customer Lifetime Value vs Acquisition Cost by Channel", pad=15)
ax.set_xlabel("Customer Acquisition Cost ($)")
ax.set_ylabel("Customer Lifetime Value ($)")

# Legend styling
ax.legend(title="Marketing Channel", loc="upper left", frameon=True)

plt.tight_layout()

# ----------------------------
# 4. Save figure as exactly 512x512 px
# ----------------------------
plt.savefig("chart.png", dpi=64, bbox_inches="tight")
plt.close()
