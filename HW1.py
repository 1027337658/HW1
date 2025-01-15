"""
Bayesian Homework: Q1, Q2, Q3

This script answers:
Q1) "What proportion of students who answer this question quickly will pass the class?"
Q2) "Given a Multinomial likelihood and a Dirichlet prior, what is the posterior distribution?"
Q3) "Create a sample from a 3-category Multinomial distribution, visualize it, then
     visualize the Dirichlet prior and posterior on a ternary plot."
     
Requires: numpy, matplotlib, scipy, plotly
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multinomial, dirichlet
import plotly.figure_factory as ff

# -------------------------------------------------------------
# Q1 
# -------------------------------------------------------------
def pass_given_quick():
    """
    Basic Bayes for Pr(pass|quick).
    """
    p_pass = 0.9
    p_q_given_pass = 0.6
    p_q_given_fail = 0.3
    p_fail = 1 - p_pass
    p_quick = p_q_given_pass * p_pass + p_q_given_fail * p_fail
    return (p_q_given_pass * p_pass) / p_quick

ans_q1 = pass_given_quick()
print("Q1: pass | quick =", round(ans_q1, 4))

# -------------------------------------------------------------
# Q2 
# -------------------------------------------------------------
def explain_conjugacy_q2():
    """
    If the prior is Dirichlet(alpha) and the likelihood is Multinomial(n,theta),
    then the posterior = Dirichlet(alpha + x).
    """
    print("[Q2] Posterior = Dirichlet(alpha + observed_counts).")

explain_conjugacy_q2()
print("Done with Q2.")
print("-" * 60)

# -------------------------------------------------------------
# Q3
# -------------------------------------------------------------
n = 50
true_p = [0.4, 0.35, 0.25]
num_samples = 5000

samples_all = multinomial.rvs(n=n, p=true_p, size=num_samples)
x1_vals = samples_all[:, 0]

plt.figure(figsize=(6,4))
plt.hist(x1_vals, bins=np.arange(-0.5, n+1, 1), density=True,
         alpha=0.7, edgecolor='black')
plt.title("Q3 Part 1: x1 from Multinomial(n=50)")
plt.xlabel("x1 count")
plt.ylabel("Rel. freq")
plt.show()

print("Q3 Part 1 done, mean x1 =", round(np.mean(x1_vals),2))
print("-" * 60)

alpha_prior = np.array([1.0, 1.0, 1.0])
d_prior = dirichlet(alpha_prior)

x_obs = np.array([20, 15, 15])
alpha_post = alpha_prior + x_obs
d_post = dirichlet(alpha_post)

def make_simplex_grid(size=100):
    """
    Make grid points in the 2-simplex for ternary plot.
    """
    pts = []
    for i in range(size+1):
        for j in range(size+1 - i):
            p1 = i / size
            p2 = j / size
            p3 = 1 - p1 - p2
            pts.append([p1,p2,p3])
    return np.array(pts)

grid_pts = make_simplex_grid()
vals_prior = d_prior.pdf(grid_pts.T)
vals_post  = d_post.pdf(grid_pts.T)

fig_pri = ff.create_ternary_contour(grid_pts.T, vals_prior,
    pole_labels=['p1','p2','p3'], interp_mode='cartesian', showscale=True)
fig_pri.update_layout(title="Q3: Dirichlet(1,1,1) prior")
fig_pri.show("png")

fig_pos = ff.create_ternary_contour(grid_pts.T, vals_post,
    pole_labels=['p1','p2','p3'], interp_mode='cartesian', showscale=True)
fig_pos.update_layout(title=f"Q3: Dirichlet({alpha_post.tolist()}) posterior")
fig_pos.show("png")

print("Q3 Part 2 finished. Observed x_obs =", x_obs, 
      "-> alpha_post =", alpha_post.tolist())
print("Check the ternary plots for how the posterior is shifted.\n")