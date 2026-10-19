# Equal gradient mean and variance do not fix the first AdamW update

Synthetic algebraic clarification, not a new general mathematical theorem and not empirical identification of SGO's mechanism. Objective/estimator interactions must not automatically be called gradient-variance-only effects.

## Assumptions and variable table

Use scalar AdamW with initial parameter and both moment buffers zero, ordinary bias correction, no AMSGrad, positive learning rate and epsilon. PyTorch 2.10 definitions: [AdamW](https://docs.pytorch.org/docs/2.10/generated/torch.optim.AdamW.html), [SGD](https://docs.pytorch.org/docs/2.10/generated/torch.optim.SGD.html). These specify the algorithms; the following algebra proves this particular counterexample.

| Symbol | Meaning (Japanese) | SI unit | Definition | Domain / assumptions | Type |
|---|---|---|---|---|---|
| A,B | 勾配の確率法則 | 1 | finite laws below | not performance arms | categorical labels |
| g | 確率勾配 | 1 | derivative of dimensionless scalar loss w.r.t. dimensionless parameter | finite real atoms | random real scalar |
| p | 勾配値の確率 | 1 | mass at an atom | nonnegative, sums to1 | probability scalar |
| E,Var | 期待値・分散演算 | 1 | probability-weighted sum; Var(g)=E[g²]−E[g]² | finite support | scalar-valued operators |
| theta_0,theta_1 | 更新前後のパラメータ | 1 | theta_0=0; theta_1 after first step | dimensionless coordinate | real scalars |
| m_0,v_0 | 更新前のモーメント状態 | 1 | both zero | fresh optimizer | real scalars |
| m_1,v_1 | 更新後のモーメント状態 | 1 | recurrences below | first step | real scalars |
| beta_1,beta_2 | モーメント減衰率 | 1 | .9,.999 in the check | 0<=beta_i<1 | real scalars |
| mhat_1,vhat_1 | バイアス補正後の状態 | 1 | m_1/(1−beta_1),v_1/(1−beta_2) | denominators positive | real scalars |
| eta | 学習率 | 1 | .005 in the check | positive | real scalar |
| eps | 分母の正則化定数 | 1 | 1e−8 in the check | positive | real scalar |
| lambda | 重み減衰率 | 1 | .001 in the check | term vanishes at theta_0=0 | real scalar |
| u(g) | パラメータ変化 | 1 | theta_1−theta_0 at gradient g | first step only | real scalar |
| f_A,f_B | 正規化期待降下量 | 1 | −E_A[u]/eta,−E_B[u]/eta | formulas below | real scalars |

Unit check: coordinates and constructed losses are dimensionless. The second-moment buffer has squared-gradient units1; its square root and eps have equal units1; u has parameter units1. This convention does not assert coordinate invariance of arbitrary neural-network optimizers.

## Exact equal moments

A takes gradients −1 and3 with probability1/2 each. B takes gradients1/3 and7 with probabilities9/10 and1/10.

E_A[g]=−1/2+3/2=1.

E_A[g²]=1/2+9/2=5.

E_B[g]=(9/10)(1/3)+(1/10)7=3/10+7/10=1.

E_B[g²]=(9/10)(1/9)+(1/10)49=1/10+49/10=5.

Both variances are5−1²=4. These are exact rational equalities, not Monte Carlo estimates.

## First AdamW step

Zero initial buffers give m_1=(1−beta_1)g and v_1=(1−beta_2)g². Dividing by the first-step bias corrections gives mhat_1=g and vhat_1=g². Decoupled decay changes the zero parameter by −eta*lambda*theta_0=0. Consequently,

u(g)=−eta*g/(sqrt(g²)+eps)=−eta*g/(abs(g)+eps).

For A:

f_A=[−1/(1+eps)+3/(3+eps)]/2

=[−(3+eps)+3(1+eps)]/[2(1+eps)(3+eps)]

=eps/[(1+eps)(3+eps)].

For B:

f_B=(9/10)*(1/3)/(1/3+eps)+(1/10)*7/(7+eps)

=(9/10)/(1+3eps)+(7/10)/(7+eps).

For0<eps<=.01, f_A<=eps/3<=1/300. Meanwhile f_B>=(9/10)/1.03=90/103>1/300, even ignoring its positive second term. Thus f_B>f_A. Since eta>0, E_B[u]<E_A[u]. This proves different expected updates despite exactly identical first two moments, without setting epsilon to zero.

## Executable numerical check

At eta=.005,eps=1e−8:

| Law | Exact expected first AdamW update | Expected first SGD update |
|---|---:|---:|
| A | −1.6666666444444447e−11 | −.005 |
| B | −.004999999864285718 | −.005 |

The executable enumerates every atom with a fresh optimizer and probability-weights the actual updates. There is no sampling error. Maximum absolute formula discrepancy over AdamW/SGD was8.673617379884035e−19 in float64 and1.1175870905794083e−10 in float32. Float32 A's AdamW expectation rounded to0; its exact value is not0.

Runtime: Python3.13.5,torch2.10.0+cpu,one CPU thread,foreach=False,fused=False,no quantization. One scalar gradient atom per call; clock uncontrolled and no throughput claim. This is a synthetic check, not one of the registered Python3.12 image-training studies.

```bash
python experiments/check_adamw_moments_counterexample.py --out new_counterexample.json
```

## Limits and ERROR CHECK

Matching only gradient mean and variance cannot generally force the same expected AdamW update. Sign probabilities and other distributional properties differ here. The SGD control illustrates the distinction between linear and adaptive first updates. This does not prove the mechanism of SGO accuracy differences, universal usefulness of noise, or an optimizer ranking. Exact rational moments, positive epsilon, bias correction, zero-parameter weight decay and actual float64/float32 PyTorch updates were checked. The explicit failure of a variance-only explanation is a logical counterexample, not another accuracy result.
