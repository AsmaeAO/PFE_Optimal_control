

using Plots

using Ipopt
using JuMP


# Coefficients du modèle
const A = 10.0
const B = 1.0
const C = 3.0
const T = 20

# Valeurs initiales
const s0 = 0.2
const i0 = 0.8
const m0 = 0.0

# Valeurs initiales 
#const p_s_0 = -6.3486
#const p_i_0 = -7.654e+1
#const p_m_0 = -6.2821e-2

# Autres paramètres
const N = 10000
const ε = 1e-9
const dt = 1/N

# Valeurs des paramètres SIMR
const b = 0.01
const c = 1.1
const a1 = 0.08
const a2 = 0.005
const g1 = 0.02
const g2 = 0.5
const η = 0.8

# Valeurs maximales des contrôles
const umax = 0.8
const vmax = 0.8
const u_sing = 0.188

# Switching points
const t1_u = 4.45
const t2_u = 13.55
const t1_v = 6.49


# Définir le modèle d'optimisation
model = Model(Ipopt.Optimizer)

# Variables d'état s(t), i(t), et m(t)
@variable(model, s[t=0:N] >= 0)
@variable(model, i[t=0:N] >= 0)
@variable(model, m[t=0:N] >= 0)

# Contraintes initiales
@constraint(model, s[0] == s0)
@constraint(model, i[0] == i0)
@constraint(model, m[0] == m0)

# Contrainte finale sur i(T)
@constraint(model, i[T] <= 5e-4)

# Définition de u_* et v_*
function u(t)

    if 0 <= t <= t1_u
        return umax
    elseif t1_u <= t <= t2_u
        return u_sing
    elseif t2_u <= t <= T
        return 0
    end
end

function v(t)
    if 0 <= t <= t1_v
        return vmax
    elseif t1_v <= t <= T
        return 0
    end
end

# Expression pour J1(u, v)
@expression(model, J1, sum((A*i[t] + B*u(t) + C*v(t)) for t in 0:T))

# Équations différentielles
for t in 0:T-1
    @constraint(model, s[t+1] == s[t] + dt * (b - b*s[t] - c*s[t]*i[t] + (a1*i[t] + a2*m[t])*s[t] - η*u(t)*s[t]))
    @constraint(model, i[t+1] == i[t] + dt * (c*s[t]*i[t] - b*i[t] - (g1 + a1)*i[t] + (a1*i[t] + a2*m[t])*i[t] - i[t]*v(t)))
    @constraint(model, m[t+1] == m[t] + dt * (-(a2 + g2 + b)*m[t] + (a1*i[t] + a2*m[t])*m[t] + i[t]*v(t)))
end

# Objectif de minimisation
@objective(model, Min, J1)

# Résoudre le problème d'optimisation
optimize!(model)

# Afficher les résultats
println("Valeur optimale de J1(u, v) : ", objective_value(model))
println("Valeurs optimales de s(T) : ", [value(s[T])] )
println("Valeurs optimales de i(T) : ", [value(i[T])] )
println("Valeurs optimales de m(T) : ", [value(m[T])] )

t_values = 0:0.1:T  # Plage de valeurs de t
u_values = [u(t) for t in t_values]
plot(t_values, u_values, xlabel="t", ylabel="u(t)", label="u(t)", legend=true)

v_values = [v(t) for t in t_values]
plot(t_values, v_values, xlabel="t", ylabel="v(t)", label="v(t)", legend=true)
