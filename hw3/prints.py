
print(
    f'------------------- PROBLEM 2 --------------------\
    --------------------------------------------------\
    {prob2.status}\
    Minimum Generating Cost : {prob2.value:4.2f} USD\
    \
    Node 0 [Grid]  Gen Power : p_0 = {p[0].value:1.3f} MW | q_0 = {q[0].value:1.3f} MW | s_0 = ({s[0].value:1.3f} MW\
    Node 3 [Gas]   Gen Power : p_3 = {p[3].value:1.3f} MW | q_3 = {q[3].value:1.3f} MW | s_3 = ({s[3].value:1.3f} MW\
    Node 9 [Solar] Gen Power : p_9 = {p[9].value:1.3f} MW | q_9 = {q[9].value:1.3f} MW | s_9 = ({s[9].value:1.3f} MW\
    \
    Total active power   : {sum(l_P):1.3f} MW   consumed | {sum(p.value):1.3f} MW   generated\
    Total reactive power : {sum(l_Q):1.3f} MVAr consumed | {sum(q.value):1.3f} MVAr generated\
    Total apparent power : {sum(l_S):1.3f} MVA  consumed | {sum(s.value):1.3f} MVA  generated\
')

print(
    f'------------------- PROBLEM 3 --------------------\
    --------------------------------------------------\
    {prob3.status}\
    Minimum Generating Cost : {prob3.value:4.2f} USD\
    \
    Node 0 [Grid]  Gen Power : p_0 = {p[0].value:1.3f} MW | q_0 = {q[0].value:1.3f} MW | s_0 = {s[0].value:1.3f}) MW || mu_s0 = {constraints[0].dual_value[0]:3.0f} USD/MW\
    Node 3 [Gas]   Gen Power : p_3 = {p[3].value:1.3f} MW | q_3 = {q[3].value:1.3f} MW | s_3 = {s[3].value:1.3f}) MW || mu_s4 = {constraints[0].dual_value[3]:3.0f} USD/MW\
    Node 9 [Solar] Gen Power : p_9 = {p[9].value:1.3f} MW | q_9 = {q[9].value:1.3f} MW | s_9 = {s[9].value:1.3f}) MW || mu_s9 = {constraints[0].dual_value[9]:3.0f} USD/MW\
    \
    Total active power   : {sum(l_P):1.3f} MW   consumed | {sum(p.value):1.3f} MW   generated\
    Total reactive power : {sum(l_Q):1.3f} MVAr consumed | {sum(q.value):1.3f} MVAr generated\
    Total apparent power : {sum(l_S):1.3f} MVA  consumed | {sum(s.value):1.3f} MVA  generated\
')

# Output Results
print(
    f'------------------- PROBLEM 4 --------------------\
    --------------------------------------------------\
    {prob4.status}\
    Minimum Generating Cost : {prob4.value:4.2f} USD\
    \
    Node 0 [Grid]  Gen Power : p_0 = {p[0].value:1.3f} MW | q_0 = {q[0].value:1.3f} MW | s_0 = {s[0].value:1.3f} MW || mu_s0 = {constraints[0].dual_value[0]:3.0f} USD/MW\
    Node 3 [Gas]   Gen Power : p_3 = {p[3].value:1.3f} MW | q_3 = {q[3].value:1.3f} MW | s_3 = {s[3].value:1.3f} MW || mu_s4 = {constraints[0].dual_value[3]:3.0f} USD/MW\
    Node 9 [Solar] Gen Power : p_9 = {p[9].value:1.3f} MW | q_9 = {q[9].value:1.3f} MW | s_9 = {s[9].value:1.3f} MW || mu_s9 = {constraints[0].dual_value[9]:3.0f} USD/MW\
    \
    Total active power   : {sum(l_P):1.3f} MW   consumed | {sum(p.value):1.3f}MW   generated\
    Total reactive power : {sum(l_Q):1.3f} MVAr consumed | {sum(q.value):1.3f}MVAr generated\
    Total apparent power : {sum(l_S):1.3f} MVA  consumed | {sum(s.value):1.3f}MVA  generated\
')