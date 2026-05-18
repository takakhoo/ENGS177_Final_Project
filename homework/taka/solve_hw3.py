import numpy as np

print("=" * 60)
print("PROBLEM 1: Inventory DP (Backward Induction)")
print("=" * 60)

T = 5
h = 1
c = 3
s0 = 5
d = [2, 3, 1, 4, 2]  # demand for days 1..5
max_inv = 13

V = np.full((T + 2, max_inv + 1), np.inf)
policy = np.full((T + 1, max_inv + 1), -1, dtype=int)

V[T + 1, :] = 0

for t in range(T, 0, -1):
    dt = d[t - 1]
    for s in range(max_inv + 1):
        best_cost = np.inf
        best_a = 0
        a_min = max(0, dt - s)
        a_max = max_inv - s
        for a in range(a_min, a_max + 1):
            s_next = s + a - dt
            if s_next < 0 or s_next > max_inv:
                continue
            cost_t = c * a + h * (s + a - dt)
            total = cost_t + V[t + 1, s_next]
            if total < best_cost:
                best_cost = total
                best_a = a
        V[t, s] = best_cost
        policy[t, s] = best_a

print(f"Optimal total cost from s_1={s0}: V_1({s0}) = {V[1, s0]}")
print()

print("Value function table:")
for t in range(1, T + 2):
    vals = [f"V_{t}({s})={V[t,s]:.0f}" for s in range(8) if V[t, s] < np.inf]
    print(f"  Day {t}: " + ", ".join(vals[:8]))
print()

print("Optimal policy trace:")
s = s0
total_cost = 0
for t in range(1, T + 1):
    a = policy[t, s]
    dt = d[t - 1]
    holding = s + a - dt
    day_cost = c * a + h * holding
    total_cost += day_cost
    print(f"  Day {t}: s={s}, d={dt}, order a*={a}, "
          f"ending_inv={holding}, cost={c}*{a}+{h}*{holding}={day_cost}, "
          f"s_next={holding}")
    s = holding
print(f"  Total cost: {total_cost}")

print()
print("Detailed V table for LaTeX:")
for t in range(1, T + 2):
    row = []
    for s in range(10):
        if V[t, s] < np.inf:
            row.append(f"{V[t,s]:.0f}")
        else:
            row.append("--")
    print(f"  t={t}: s=0..9 -> {row}")

print()
print("=" * 60)
print("PROBLEM 2: MDP")
print("=" * 60)

P = np.array([
    [0.3, 0.4, 0.2, 0.1],
    [0.2, 0.3, 0.5, 0.0],
    [0.1, 0.0, 0.8, 0.1],
    [0.4, 0.0, 0.0, 0.6]
])

r_stay = np.array([1.0, 2.0, 3.0, 4.0])
gamma_leave = 20.0
lam = 0.9
n_states = 4

print("\n--- (a) MDP Formulation ---")
print("States: {s1, s2, s3, s4} + absorbing state s0")
print("Actions: {stay, leave}")
print(f"Reward(si, stay) = i, Reward(si, leave) = {gamma_leave}")
print(f"Discount factor lambda = {lam}")
print(f"Transition: stay -> P, leave -> absorbing s0")

print("\n--- (b) Modified Policy Iteration ---")
eps = 0.01
v = np.zeros(n_states)
n = 0
threshold = eps * (1 - lam) / (2 * lam)
print(f"epsilon = {eps}, threshold = eps*(1-lam)/(2*lam) = {threshold:.6f}")

max_outer = 100
for n in range(max_outer):
    m_n = n + 10

    q_stay = r_stay + lam * P @ v
    q_leave = np.full(n_states, gamma_leave)

    pi = np.array(['stay' if q_stay[i] >= q_leave[i] else 'leave' for i in range(n_states)])

    u = np.maximum(q_stay, q_leave)

    span = np.max(u - v) - np.min(u - v)
    diff = np.max(np.abs(u - v))

    print(f"\n  Iteration n={n}: m_n={m_n}")
    print(f"    v^n = [{', '.join(f'{x:.4f}' for x in v)}]")
    print(f"    Q_stay = [{', '.join(f'{x:.4f}' for x in q_stay)}]")
    print(f"    Q_leave = [{', '.join(f'{x:.4f}' for x in q_leave)}]")
    print(f"    pi_{n+1} = {list(pi)}")
    print(f"    u_n^0 = [{', '.join(f'{x:.4f}' for x in u)}]")
    print(f"    ||u_n^0 - v^n||_inf = {diff:.6f}")

    if diff < threshold:
        print(f"    CONVERGED: {diff:.6f} < {threshold:.6f}")
        v = u
        final_pi = pi
        break

    r_pi = np.array([r_stay[i] if pi[i] == 'stay' else gamma_leave for i in range(n_states)])
    P_pi = np.zeros((n_states, n_states))
    for i in range(n_states):
        if pi[i] == 'stay':
            P_pi[i] = P[i]

    for k in range(m_n):
        u = r_pi + lam * P_pi @ u

    v = u
    print(f"    After {m_n} partial eval steps:")
    print(f"    v^{n+1} = [{', '.join(f'{x:.4f}' for x in v)}]")

final_pi_mpi = pi
final_v_mpi = v
print(f"\n  RESULT: pi_eps = {list(final_pi_mpi)}")
print(f"  RESULT: v_eps = [{', '.join(f'{x:.4f}' for x in final_v_mpi)}]")

print("\n--- (c) Policy Iteration ---")

pi = np.array(['leave'] * n_states)
iteration = 0

for iteration in range(20):
    r_pi = np.array([r_stay[i] if pi[i] == 'stay' else gamma_leave for i in range(n_states)])
    P_pi = np.zeros((n_states, n_states))
    for i in range(n_states):
        if pi[i] == 'stay':
            P_pi[i] = P[i]

    A = np.eye(n_states) - lam * P_pi
    v_pi = np.linalg.solve(A, r_pi)

    print(f"\n  PI Iteration {iteration}:")
    print(f"    pi = {list(pi)}")
    print(f"    r_pi = [{', '.join(f'{x:.1f}' for x in r_pi)}]")
    print(f"    v_pi = [{', '.join(f'{x:.4f}' for x in v_pi)}]")

    q_stay = r_stay + lam * P @ v_pi
    q_leave = np.full(n_states, gamma_leave)
    new_pi = np.array(['stay' if q_stay[i] >= q_leave[i] else 'leave' for i in range(n_states)])

    print(f"    Q_stay = [{', '.join(f'{x:.4f}' for x in q_stay)}]")
    print(f"    Q_leave = [{', '.join(f'{x:.4f}' for x in q_leave)}]")
    print(f"    new_pi = {list(new_pi)}")

    if np.all(new_pi == pi):
        print(f"\n  POLICY ITERATION CONVERGED at iteration {iteration}")
        print(f"  Optimal policy: {list(pi)}")
        print(f"  Optimal values: [{', '.join(f'{x:.4f}' for x in v_pi)}]")
        final_v_pi = v_pi
        final_pi_pi = pi
        break
    pi = new_pi

print("\n  Verification: (I - lam*P) * v = r under all-stay policy")
A_check = np.eye(4) - lam * P
print(f"  A*v = {A_check @ final_v_pi}")
print(f"  r   = {r_stay}")

print("\n--- (d) Smallest gamma for leaving in state 2 ---")

print("  Under all-stay policy, v*(s2) determines the threshold.")
print(f"  v*(s2) = {final_v_pi[1]:.4f}")
print(f"  At gamma = v*(s2), the agent is indifferent between stay and leave in s2.")

for gamma_test in np.arange(15, 30, 0.01):
    q_stay_test = r_stay + lam * P @ final_v_pi
    if gamma_test >= q_stay_test[1]:
        print(f"  Searching: gamma >= Q_stay(s2) = {q_stay_test[1]:.4f}")
        break

v_all_stay = np.linalg.solve(np.eye(4) - lam * P, r_stay)
print(f"\n  Value under all-stay: {v_all_stay}")
print(f"  For s2: v(s2) = {v_all_stay[1]:.4f}")
print(f"  gamma_threshold = v(s2) = {v_all_stay[1]:.4f}")

print("\n  Verification by parametric analysis:")
print("  We need to find gamma such that gamma = 2 + 0.9 * sum_j P(s2,j)*v(sj)")
print("  where v is the value under the optimal policy that might change with gamma.")
print()
print("  If the policy is 'stay everywhere', v doesn't depend on gamma,")
print("  so gamma_threshold = v_stay(s2).")
print()

q2_stay = r_stay[1] + lam * np.dot(P[1], v_all_stay)
print(f"  Q(s2, stay) = 2 + 0.9 * P[s2,:] . v = {q2_stay:.4f}")
print(f"  Smallest gamma for leaving s2 = {q2_stay:.4f}")

print("\n  But we should check: does the policy change in other states first?")
q_all = r_stay + lam * P @ v_all_stay
print(f"  Q(si, stay) under all-stay values: {q_all}")
print(f"  For leaving to be optimal in si: gamma > Q(si, stay)")
print(f"  s1: gamma > {q_all[0]:.4f}")
print(f"  s2: gamma > {q_all[1]:.4f}")
print(f"  s3: gamma > {q_all[2]:.4f}")
print(f"  s4: gamma > {q_all[3]:.4f}")
print(f"\n  Smallest gamma for leaving s2 (under all-stay policy) = {q_all[1]:.4f}")

print("\n  Note: v(si) = Q(si, stay) under all-stay since all-stay IS the policy.")
print(f"  Confirming: v_all_stay = {v_all_stay}")
print(f"  The answer is gamma* = v*(s2) = {v_all_stay[1]:.4f}")

print()
print("=" * 60)
print("PROBLEM 3: epsilon-greedy Bandit")
print("=" * 60)

arms_chosen = [1, 2, 2, 2, 3]
rewards = [1, 1, 2, 2, 0]
n_arms = 4

Q = {a: 0.0 for a in range(1, n_arms + 1)}
N = {a: 0 for a in range(1, n_arms + 1)}

print("\nStep-by-step Q-value updates:")
print(f"  Initial: Q = {Q}")

for t in range(1, 6):
    a_t = arms_chosen[t - 1]
    r_t = rewards[t - 1]

    greedy_val = max(Q.values())
    greedy_arms = [a for a in range(1, n_arms + 1) if Q[a] == greedy_val]

    print(f"\n  t={t}: Q_t = {{{', '.join(f'{k}:{v:.4f}' for k, v in Q.items())}}}")
    print(f"    Greedy arms (argmax Q): {greedy_arms} (value={greedy_val:.4f})")
    print(f"    Chosen arm: {a_t}, Reward: {r_t}")

    if a_t in greedy_arms:
        print(f"    -> Chosen arm IS a greedy arm -> could be exploitation OR exploration")
        must_explore = False
    else:
        print(f"    -> Chosen arm is NOT a greedy arm -> MUST be exploration")
        must_explore = True

    N[a_t] += 1
    Q[a_t] = Q[a_t] + (r_t - Q[a_t]) / N[a_t]

    print(f"    After update: Q = {{{', '.join(f'{k}:{v:.4f}' for k, v in Q.items())}}}, "
          f"N = {dict(N)}")

print("\n\nSummary:")
print("  (a) MUST be exploration: t=2 (arm 2 chosen, only arm 1 is greedy)")
print("                          t=5 (arm 3 chosen, only arm 2 is greedy)")
print("  (b) COULD be exploration: t=1 (all arms tied, any is greedy)")
print("                            t=3 (arms 1,2 tied, arm 2 is greedy)")
print("                            t=4 (arm 2 uniquely greedy, arm 2 chosen)")
