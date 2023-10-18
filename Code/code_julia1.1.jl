

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

# Définir le modèle d'optimisation
model = Model(Ipopt.Optimizer)


# # Variables de contrôle u(t) et v(t)
# @variable(model, 0 <= u[t=0:N] <= umax)
# @variable(model, 0 <= v[t=0:N] <= vmax)


# Variables d'état s(t), i(t), et m(t)
@variable(model, s[t=0:N] >= 0)
@variable(model, i[t=0:N] >= 0)
@variable(model, m[t=0:N] >= 0)


# Variables d'état adjoint ps(t), pi(t), pm(t)
#@variable(model, p_s[t=0:N] >= 0)
#@variable(model, p_i[t=0:N] >= 0)
#@variable(model, p_m[t=0:N] >= 0)

# Contraintes initiales
@constraint(model, s[0] == s0)
@constraint(model, i[0] == i0)
@constraint(model, m[0] == m0)


# Contraintes initiales états adjoints
# ces valeurs sont les valeurs numériques obtenues, on les prend quand meme pour voir si ca peut marcher 

#@constraint(model, p_s[0] == p_s_0)
#@constraint(model, p_i[0] == p_i_0)
#@constraint(model, p_m[0] == p_m_0)


# Contrainte finale sur i(T)
@constraint(model, i[T] <= 5e-4)
#@constraint(model, p_s[T] == 0)
#@constraint(model, p_i[T] >= 0)
#@constraint(model, p_m[T] == 0)

# Définition de phi_1 et phi_2 pour en déduire u_* et v_*
function phi_1(t)
    return -η * p_s(t) * s(t) - B
end

function phi_2(t)
    return (p_m(t) - p_i(t)) * i(t) - C
end
    
function u(t)
    phi1_value = -η * p_s(t) * s(t) - B

    if phi1_value > 0
        return umax
    elseif phi1_value == 0
        return u_sing
    else
        return 0
    end
end

function v(t)
    phi2_value = (p_m(t) - p_i(t)) * i(t) - C

    if phi2_value > 0
        return vmax
    elseif phi_2_value == 0
        return error
    else
        return 0
    end
end

# Expression pour J1(u, v)
@expression(model, J1, sum((A*i[t] + B*u(t) + C*v(t)) for t in 0:T))

# Équations différentielles
for t in 0:T-1
    @constraint(model, s[t+1] == s[t] + dt * (b - b*s[t] - c*s[t]*i[t] + (a1*i[t] + a2*m[t])*s[t] - η*u[t]*s[t]))
    @constraint(model, i[t+1] == i[t] + dt * (c*s[t]*i[t] - b*i[t] - (g1 + a1)*i[t] + (a1*i[t] + a2*m[t])*i[t] - i[t]*v[t]))
    @constraint(model, m[t+1] == m[t] + dt * (-(a2 + g2 + b)*m[t] + (a1*i[t] + a2*m[t])*m[t] + i[t]*v[t]))
    #@constraint(model, p_s[t+1] == p_s[t] - dt * [(-b-c*i[t] +a1*i[t] + a2*m[t] - η*u[t])p_s[t] + c*i[t]*p_i[t]])
    #@constraint(model, p_i[t+1] == p_i[t] - dt * [(-c+a1)*s[t]*p_s[t] + (c*s[t] - a1 - g1 -b +a1*i[t] +a2*m[t] -v[t])*p_i[t] + a1*i[t]*p_i[t] + (a1*m[t]* + v[t])*p_m[t] - A ] )
    #@constraint(model, p_m[t+1] == p_m[t] - dt * [a2*s[t]*p_s[t] + a2*i[t]*p_i[t] +(a1*i[t] + 2*a2*m[t] -a2 -g2-b)*p_m[t]])
end

# Objectif de minimisation
@objective(model, Min, J1)

# Résoudre le problème d'optimisation
optimize!(model)

# Afficher les résultats
println("Valeur optimale de J1(u, v) : ", objective_value(model))
println("Valeurs optimales de u(T) : ", [value(u[T])] )
println("Valeurs optimales de v(T) : ", [value(v[T])] )
println("Valeurs optimales de s(T) : ", [value(s[T])] )
println("Valeurs optimales de i(T) : ", [value(i[T])] )
println("Valeurs optimales de m(T) : ", [value(m[T])] )
