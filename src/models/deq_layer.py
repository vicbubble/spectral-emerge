import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

def anderson_solver(f, x0, context, max_iter, tol, m=5):
    """Anderson acceleration solver for fixed point iteration."""
    with torch.no_grad():
        x = x0
        X = [x]
        F = [f(x, context)]
        
        x = F[0]
        X.append(x)
        F.append(f(x, context))
        
        x = F[0]
        X.append(x)
        F.append(f(x, context))
        
        for k in range(2, max_iter):
            res_norm = torch.norm(F[-1] - X[-1], dim=-1).max().item()
            if res_norm < tol:
                return X[-1], k, res_norm
                
            m_k = min(len(X)-1, m)
            dX = torch.stack([X[-i] - X[-i-1] for i in range(1, m_k+1)], dim=-1)
            dF = torch.stack([F[-i] - F[-i-1] for i in range(1, m_k+1)], dim=-1)
            
            dG = dF - dX
            dG_T = dG.transpose(1, 2)
            A = torch.bmm(dG_T, dG) + 1e-4 * torch.eye(m_k, device=x.device).unsqueeze(0)
            B = torch.bmm(dG_T, (F[-1] - X[-1]).unsqueeze(-1))
            
            alpha = torch.linalg.solve(A, B).squeeze(-1)
            
            x_next = F[-1] - (dF * alpha.unsqueeze(1)).sum(-1)
            x = x_next
            X.append(x)
            F.append(f(x, context))
            if len(X) > m + 1:
                X.pop(0)
                F.pop(0)
                
        res_norm = torch.norm(F[-1] - X[-1], dim=-1).max().item()
        return X[-1], max_iter, res_norm

class DEQImplicitDiff(torch.autograd.Function):
    """Custom Autograd function for exact implicit differentiation."""
    @staticmethod
    def forward(ctx, f, z_star, context, max_iter, tol, *params):
        ctx.f = f
        ctx.save_for_backward(z_star, context, *params)
        ctx.max_iter = max_iter
        ctx.tol = tol
        return z_star.clone()

    @staticmethod
    def backward(ctx, grad_z_star):
        z_star, context, *params = ctx.saved_tensors
        with torch.enable_grad():
            z = z_star.detach().requires_grad_(True)
            c = context.detach().requires_grad_(True)
            f_z = ctx.f(z, c)
            
        def vjp_z(v):
            return torch.autograd.grad(f_z, z, grad_outputs=v, retain_graph=True)[0]
            
        v = grad_z_star
        for _ in range(ctx.max_iter):
            v_next = grad_z_star + vjp_z(v)
            if torch.norm(v_next - v).max() < ctx.tol:
                v = v_next
                break
            v = v_next
            
        grad_context, *grad_params = torch.autograd.grad(f_z, (c,) + tuple(params), grad_outputs=v, retain_graph=False)
        return (None, None, grad_context, None, None) + tuple(grad_params)

class DEQLayer(nn.Module):
    """Deep Equilibrium Layer with Anderson Acceleration and Implicit Diff."""
    def __init__(self, x_dim, latent_dim, hidden_dim, deq_max_iter=50, deq_tol=1e-4):
        super().__init__()
        self.max_iter = deq_max_iter
        self.tol = deq_tol
        self.latent_dim = latent_dim
        
        self.f_theta = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim + x_dim, hidden_dim)),
            nn.Tanh(),
            spectral_norm(nn.Linear(hidden_dim, latent_dim)),
            nn.Tanh()
        )
        
    def _f(self, z, context):
        x = torch.cat([z, context], dim=-1)
        return self.f_theta(x)
        
    def forward(self, context):
        bsz = context.size(0)
        z0 = torch.zeros(bsz, self.latent_dim, device=context.device)
        
        # 1. Compute fixed point with no_grad
        z_star_fixed, n_iters, res_norm = anderson_solver(self._f, z0, context, self.max_iter, self.tol)
        
        # 2. Implicit diff using custom autograd function
        params = list(self.f_theta.parameters())
        z_star = DEQImplicitDiff.apply(self._f, z_star_fixed, context, self.max_iter, self.tol, *params)
        
        convergence_info = {'n_iters': n_iters, 'residual_norm': res_norm}
        return z_star, convergence_info
