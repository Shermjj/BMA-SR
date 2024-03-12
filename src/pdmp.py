import numpy as np
import torch
import bisect
import bisect
from scipy.stats import genextreme
from scipy.optimize import minimize_scalar
import scipy
import functools
from numpy import linalg as la
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

eps = 2.220446049250313E-016
epsi_default = np.sqrt ( eps )
t_default = 10.0 * np.sqrt ( eps )

@partial(jax.jit, static_argnums=(0))
def brent_minimize(f, a, b, tol=1e-5, maxiter=500):
    golden_ratio = 0.5 * (3.0 - jnp.sqrt(5.0))

    x = w = v = a + golden_ratio * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0

    def cond_fun(loop_vars):
        _, _, _, _, _, _, _, _, _, _, iter, converged = loop_vars
        return (iter < maxiter) & (~converged)

    def body_fun(loop_vars):
        x, w, v, fx, fw, fv, d, e, a, b, iter, converged = loop_vars
        m = 0.5 * (a + b)
        tol1 = tol * jnp.abs(x) + tol
        tol2 = 2.0 * tol1

        is_converged = jnp.abs(x - m) <= (tol2 - 0.5 * (b - a))
        converged = converged | is_converged

        is_parabolic_possible = tol1 < jnp.abs(e)
        r = (x - w) * (fx - fv)
        q = (x - v) * (fx - fw)
        p = (x - v) * q - (x - w) * r
        q = 2.0 * (q - r)
        p = jnp.where(q > 0.0, -p, p)
        q = jnp.abs(q)
        is_parabolic = jnp.logical_and(
            is_parabolic_possible,
            jnp.abs(p) < jnp.abs(0.5 * q * e)
        ) & (p > q * (a - x)) & (p < q * (b - x))

        d = jnp.where(is_parabolic, p / q, golden_ratio * (jnp.where(x < m, b - x, a - x)))
        u = jnp.where(jnp.abs(d) >= tol1, x + d, x + jnp.sign(tol1) * d)
        fu = f(u)

        not_u_lt_x = ~(u < x)
        a = jnp.where(u < x, u, a)
        b = jnp.where(not_u_lt_x, u, b)

        not_fu_le_fw = ~(fu <= fw)
        v = jnp.where(jnp.logical_and(fu <= fw, w != x), u, v)
        fv = jnp.where(jnp.logical_and(fu <= fw, w != x), fu, fv)

        w = jnp.where(jnp.logical_or(fu <= fw, w == x), u, w)
        fw = jnp.where(jnp.logical_or(fu <= fw, w == x), fu, fw)

        x = jnp.where(fu <= fx, u, x)
        fx = jnp.where(fu <= fx, fu, fx)

        e = jnp.where(is_parabolic, d, 0.0)

        iter += 1
        return x, w, v, fx, fw, fv, d, e, a, b, iter, converged

    initial_vars = (x, w, v, fx, fw, fv, d, e, a, b, 0, False)
    final_vars = jax.lax.while_loop(cond_fun, body_fun, initial_vars)

    x, _, _, fx, _, _, _, _, _, _, evals, conv = final_vars
    return x, fx, evals, conv

def local_min ( f, a, b, epsi = epsi_default, t=t_default, maxiter=50 ):

#*****************************************************************************80
#
## LOCAL_MIN seeks a local minimum of a function F(X) in an interval [A,B].
#
#  Discussion:
#
#    The method used is a combination of golden section search and
#    successive parabolic interpolation.  Convergence is never much slower
#    than that for a Fibonacci search.  If F has a continuous second
#    derivative which is positive at the minimum (which is not at A or
#    B), then convergence is superlinear, and usually of the order of
#    about 1.324....
#
#    The values EPSI and T define a tolerance TOL = EPSI * abs ( X ) + T.
#    F is never evaluated at two points closer than TOL.
#
#    If F is a unimodal function and the computed values of F are always
#    unimodal when separated by at least SQEPS * abs ( X ) + (T/3), then
#    LOCAL_MIN approximates the abscissa of the global minimum of F on the
#    interval [A,B] with an error less than 3*SQEPS*abs(LOCAL_MIN)+T.
#
#    If F is not unimodal, then LOCAL_MIN may approximate a local, but
#    perhaps non-global, minimum to the same accuracy.
#
#    Thanks to Jonathan Eggleston for pointing out a correction to the 
#    golden section step, 01 July 2013.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    03 December 2016
#
#  Author:
#
#    Original FORTRAN77 version by Richard Brent.
#    Python version by John Burkardt.
#
#  Reference:
#
#    Richard Brent,
#    Algorithms for Minimization Without Derivatives,
#    Dover, 2002,
#    ISBN: 0-486-41998-3,
#    LC: QA402.5.B74.
#
#  Parameters:
#
#    Input, real A, B, the endpoints of the interval.
#
#    Input, real EPSI, a positive relative error tolerance.
#    EPSI should be no smaller than twice the relative machine precision,
#    and preferably not much less than the square root of the relative
#    machine precision.
#
#    Input, real T, a positive absolute error tolerance.
#
#    Input, function value = F ( x ), the name of a user-supplied
#    function whose local minimum is being sought.
#
#    Output, real X, the estimated value of an abscissa
#    for which F attains a local minimum value in [A,B].
#
#    Output, real FX, the value F(X).
#

#
#  C is the square of the inverse of the golden ratio.
#
  converged = False
  c = 0.5 * ( 3.0 - np.sqrt ( 5.0 ) )

  sa = a
  sb = b
  x = sa + c * ( b - a )
  w = x
  v = w
  e = 0.0
  fx = f ( x )
  fw = fx
  fv = fw
  #maxiter = 1000
  #while ( True ):
  for iter in range(maxiter):

    #print('At iteration ='+str(iter)+'x is'+str(x)+', sa is'+str(sa)+', sb is'+str(sb))

    m = 0.5 * ( sa + sb )
    tol = epsi * abs ( x ) + t
    t2 = 2.0 * tol
#
#  Check the stopping criterion.
#
    if ( abs ( x - m ) <= t2 - 0.5 * ( sb - sa ) ):
      converged=True
      break
#
#  Fit a parabola.
#
    r = 0.0
    q = r
    p = q

    if ( tol < abs ( e ) ):

      r = ( x - w ) * ( fx - fv )
      q = ( x - v ) * ( fx - fw )
      p = ( x - v ) * q - ( x - w ) * r
      q = 2.0 * ( q - r )

      if ( 0.0 < q ):
        p = - p

      q = abs ( q )

      r = e
      e = d

    if ( abs ( p ) < abs ( 0.5 * q * r ) and \
         q * ( sa - x ) < p and \
         p < q * ( sb - x ) ):
#
#  Take the parabolic interpolation step.
#
      d = p / q
      u = x + d
#
#  F must not be evaluated too close to A or B.
#
      if ( ( u - sa ) < t2 or ( sb - u ) < t2 ):

        if ( x < m ):
          d = tol
        else:
          d = - tol
#
#  A golden-section step.
#
    else:

      if ( x < m ):
        e = sb - x
      else:
        e = sa - x

      d = c * e
#
#  F must not be evaluated too close to X.
#
    if ( tol <= abs ( d ) ):
      u = x + d
    elif ( 0.0 < d ):
      u = x + tol
    else:
      u = x - tol

    fu = f ( u )
#
#  Update A, B, V, W, and X.
#
    if ( fu <= fx ):

      if ( u < x ):
        sb = x
      else:
        sa = x

      v = w
      fv = fw
      w = x
      fw = fx
      x = u
      fx = fu

    else:

      if ( u < x ):
        sa = u
      else:
        sb = u

      if ( fu <= fw or w == x ):
        v = w
        fv = fw
        w = u
        fw = fu
      elif ( fu <= fv or v == x or v == w ):
        v = u
        fv = fu

  return x, fx, sa, sb, iter, converged


def mod_brent_corbella(upper_bound, func_to_min, eps=1e-6, round=True, jax=False):
  if jax==True:
    x, fx, evals, converged = brent_minimize(func_to_min, 0, upper_bound)
    return fx, evals
    
  if round == False:
    x, fx, sa, sb, iter, converged = local_min(func_to_min, 0, upper_bound)
    Λbar = fx
    evals = iter
  elif round == True:
    x, fx, sa, sb, iter, converged = local_min(func_to_min, 0, upper_bound, maxiter=2)
    if converged == True:
      Λbar = fx
      evals = iter
      print("converged")
    else:
      λlower = func_to_min(sa)
      λcandidate = fx
      # if (all(y->y==Optim.x_lower_trace(optimΛ)[1], Optim.x_lower_trace(optimΛ)) &&
      if (λlower <= func_to_min(sa + eps) and λlower < λcandidate):
        Λbar = λlower
        evals = iter + 1
        print("un converged close to LB")
      else:
        λupper = func_to_min(sb)
        # if (all(y->y==Optim.x_upper_trace(optimΛ)[1], Optim.x_upper_trace(optimΛ))&&
        if (λupper <= func_to_min(sb - eps) and λupper < λcandidate):
          Λbar = λupper
          evals = iter + 1
        else:
          x, fx, sa, sb, iter, converged = local_min(func_to_min, 0, upper_bound)
          Λbar = fx
          evals = iter + 1
  else:
    print("round should be either true or false")
  return Λbar, evals

class boomerang_sampler_gibbs: 
    """
    In this code we automatically compute upper bound using arXiv:2206.11410

    Parameters
    ----------
    sigma_ref : np.array    

    mu_ref : np.array

    gradient : 
    function taking in the tuple pair theta containing both gibbs and non gibbs variable
    but returns only the gradient wrt to the non gibbs variable 

    gibbs_sampler : 
    function taking in the non gibbs variable and returns the update for the non gibbs variable

    initial_pos : 
    tuple of np.array that contains the non gibbs and gibbs variables
    Decompose theta = (x, alpha) as the gibbs and non gibbs variable respectively

    niter : int

    lr : float
    """
    def __init__(self, 
                 sigma_ref, 
                 mu_ref, 
                 gradient, 
                 initial_pos, 
                 niter, 
                 lr, 
                 initial_t_max_guess, 
                 d, 
                 gibbs_sampler, 
                 update_sigma_ref = None,
                 seed=0, q=0.9, gibbs_ref = np.inf , adaptive=False):
        
        np.random.seed(seed)
        self.key_idx = seed
        self.d = d 
        self.gradient_count = 0
        self.gradient = gradient
        self.initial_pos = initial_pos
        self.niter = niter
        self.lr = lr
        self.sigma_ref = sigma_ref
        self.mu_ref = mu_ref
        self.t_max = initial_t_max_guess
        self.no_refresh = 0
        self.no_switch = 0
        self.q = q
        self.no_exceeded_bounds = 0
        self.no_refresh_events = 0
        self.no_gibbs_events = 0
        self.no_horizon_events = 0
        self.gibbs_sampler = gibbs_sampler
        self.gibbs_ref = gibbs_ref
        self.proposals = []
        self.upper_bounds = []
        self.adaptive = adaptive
        self.update_sigma = update_sigma_ref
        self.gradients = []

    def elliptic_dynamics(self, x, v, t):
        """
        Advanves the elliptic dynamics by time t using the initial position x and velocity v
        Note: x should just be the non-gibbs variable
        """
        x_t = self.mu_ref + (x - self.mu_ref) * np.cos(t) + v * np.sin(t)
        v_t = - (x - self.mu_ref) * np.sin(t) + v * np.cos(t)
        return np.array([x_t, v_t])

    def rate(self, time, theta, vel):
        """
        theta: tuple pair of gibbs and non gibbs variable (x, alpha)
        """
        key = jax.random.PRNGKey(self.key_idx)
        self.key_idx += 1
        skel = self.elliptic_dynamics(theta[0], vel, time)
        grad = self.gradient(key, (skel[0], theta[1]), mu_ref = self.mu_ref, sigma_ref=self.sigma_ref)
        return (np.dot(skel[1], grad)>0)*np.dot(skel[1], grad)

    def trajectory_sample(self, no_samples=None):
        if no_samples is None:
            no_samples = self.niter
        skel_t = np.vstack([x[0] for x in self.skeleton]).reshape(-1)
        skel_v = np.vstack([x[2] for x in self.skeleton])
        skel_x = np.vstack([x[1][0] for x in self.skeleton])

        traj_corr_samples = []
        sample_time = np.linspace(0, skel_t[-1] - 10e-5, no_samples)
        for iter_time in sample_time:
            iter_index = bisect.bisect_left(skel_t, iter_time) - 1 #t in [t_k, tK+1], index of t_k
            traj_corr_samples.append(self.elliptic_dynamics(skel_x[iter_index,:], skel_v[iter_index,:], iter_time - skel_t[iter_index])[0])

        return np.vstack(traj_corr_samples)
    
    def estimate_upper_bound(self, theta, v):
        """ Estimate upper bound using Brent's method"""
        negglobrate = lambda t: - self.rate(t, theta, v)
        #brent_min, funeval = mod_brent_corbella(self.t_max, negglobrate, eps=1e-6, round=False)
        res = minimize_scalar(negglobrate, bounds=(0, self.t_max), method='bounded',
                        options={"maxiter": 100, "disp": 0})
        brent_min = res.fun

        #self.tot_fun_eval = self.tot_fun_eval + funeval
        upper_bound = - brent_min

        if upper_bound > 500000 or upper_bound < 0.01:
            opt_t = res.x
            self.gradients.append((self.skel_count, opt_t, theta, v, self.rate(opt_t, theta, v))
                                    )
            print("WARNING: upper bound is very large/small")

        print(f"estimated UB: {upper_bound}")
        if upper_bound==0:
            print('bound became zero, adding a small perturbation')
            upper_bound = upper_bound + 10e-7

        if(np.isnan(upper_bound)):
            print("upper bound is nan")
        self.upper_bounds.append(upper_bound)

        return upper_bound
    
    def sample(self, key):
        self.skeleton = [(0, self.initial_pos, np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), size=1).reshape(-1))] # [(time, theta=(x, alpha), velocity)] list of tuples 
        self.skel_count = 1
        T, theta, v = self.skeleton[0] 
        x, alpha = theta
        upper_bound = self.estimate_upper_bound((x, alpha), v)
        tau_star = np.random.exponential(1/upper_bound) #Propose switching time
        tau_opt = tau_star #Tau optimal is the time since last upper bound optimisation
        tau_ref = np.random.exponential(self.lr) #Obtain refresh time
        tau_gibbs = np.random.exponential(self.gibbs_ref) #Obtain gibbs time

        while self.skel_count < self.niter:
            #Main loop
            u = 0 
            proposal_counter = 0
            rejection_counter = 0
            while tau_opt == np.min(np.array([self.t_max, tau_opt, tau_ref, tau_gibbs])) and u == 0:
                # Loop invariant: tau_opt is the minimum of the three
                lambda_opt = self.rate(tau_opt, (x, alpha), v)
                if(lambda_opt/upper_bound > 1):
                    self.no_exceeded_bounds += 1
                    print("No of exceeded bounds: ", self.no_exceeded_bounds / self.skel_count)
                    upper_bound = lambda_opt

                if(lambda_opt == 0):
                    #print("debug here")
                    pass

                u = np.random.binomial(1, lambda_opt/upper_bound)
                if u == 1:
                    # Accept the proposed switching time with probability lambda_opt/upper_bound
                    # Update T,theta=(x,alpha),v
                    T = T + tau_opt
                    x, v = self.elliptic_dynamics(x, v, tau_opt)
                    key = jax.random.PRNGKey(self.key_idx)
                    self.key_idx += 1
                    grad_x = self.gradient(key, (x,alpha), mu_ref = self.mu_ref, sigma_ref=self.sigma_ref)
                    v = v - 2*np.dot(((np.dot(v,grad_x))/(np.linalg.norm(np.dot(np.linalg.cholesky(self.sigma_ref),grad_x))**2)),np.dot(self.sigma_ref,grad_x))
                    self.skeleton.append((T, (x, alpha), v)) #Update skeleton
                    self.skel_count += 1
                    print(f"Switch Event :{self.skel_count} at time {T}")

                    # Update upper bound
                    upper_bound = self.estimate_upper_bound((x, alpha), v)
                    tau_star = np.random.exponential(1/upper_bound) #Propose switching time
                    tau_opt = tau_star
                    tau_ref = np.random.exponential(self.lr) #Obtain refresh time
                    tau_gibbs = np.random.exponential(self.gibbs_ref) #Obtain gibbs time
                else:
                    rejection_counter += 1
                    if rejection_counter > 100:
                        # Do our heuristic
                        u = 1 #Set u to 1 to "accept" the refresh time

                        self.no_refresh_events += 1
                        T = T + tau_opt
                        x, v = self.elliptic_dynamics(x, v, tau_opt)[0], np.random.multivariate_normal(np.zeros(self.d), self.sigma_ref)
                        self.skeleton.append((T, (x, alpha), v)) #Update skeleton
                        self.skel_count += 1
                        print(self.skel_count)
                        print(f"(Force) Refresh Event : {x}")

                        # Update upper bound
                        upper_bound = self.estimate_upper_bound((x, alpha), v)
                        tau_star = np.random.exponential(1/upper_bound) #Propose switching time
                        tau_opt = tau_star
                        tau_ref = np.random.exponential(self.lr) #Obtain refresh time
                        tau_gibbs = np.random.exponential(self.gibbs_ref) #Obtain gibbs time
                    else:
                        # Reject the proposed switching time
                        tau_star = np.random.exponential(1/upper_bound)
                        tau_opt = tau_opt + tau_star

                proposal_counter += 1

            self.proposals.append(proposal_counter)

            #Checking u=0 to enforce that if skeleton point was accepted in switch event, wait until next loop 
            if tau_ref == np.min(np.array([self.t_max, tau_opt, tau_ref, tau_gibbs])) and u == 0:
                #Refreshment time is reached
                self.no_refresh_events += 1
                T = T + tau_ref
                x, v = self.elliptic_dynamics(x, v, tau_ref)[0], np.random.multivariate_normal(np.zeros(self.d), self.sigma_ref)
                self.skeleton.append((T, (x, alpha), v)) #Update skeleton
                self.skel_count += 1
                print(self.skel_count)
                print(f"Refresh Event : {x} at time {T}")

                # Update upper bound
                upper_bound = self.estimate_upper_bound((x, alpha), v)
                tau_star = np.random.exponential(1/upper_bound) #Propose switching time
                tau_opt = tau_star
                tau_ref = np.random.exponential(self.lr) #Obtain refresh time
                tau_gibbs = np.random.exponential(self.gibbs_ref) #Obtain gibbs time

            if self.t_max == np.min(np.array([self.t_max, tau_opt, tau_ref, tau_gibbs])) and u == 0:
                # Horizon is reached
                self.no_horizon_events += 1
                T = T + self.t_max
                x, v = self.elliptic_dynamics(x, v, self.t_max)
                print(f"Horizon Event : {x} at time {T}")

                # Update upper bound
                upper_bound = self.estimate_upper_bound((x, alpha), v)
                tau_star = np.random.exponential(1/upper_bound) #Propose switching time
                tau_opt = tau_star
                tau_ref = np.random.exponential(self.lr) #Obtain refresh time
                tau_gibbs = np.random.exponential(self.gibbs_ref) #Obtain gibbs time

            
            if tau_gibbs == np.min(np.array([self.t_max, tau_opt, tau_ref, tau_gibbs])) and u == 0:
                #Gibbs Step
                self.no_gibbs_events += 1
                self.key_idx += 1
                key = jax.random.PRNGKey(self.key_idx)
                alpha = self.gibbs_sampler(key, (x, alpha))
                T = T + tau_gibbs
                x, v = self.elliptic_dynamics(x, v, tau_gibbs)

                # Update the sigma_ref if we use adaptive preconditioning, otherwise don't 
                if self.adaptive:
                    self.sigma_ref = self.update_sigma(alpha)

                self.skeleton.append((T, (x, alpha), v)) #Update skeleton
                self.skel_count += 1
                print(self.skel_count)
                print(f"Gibbs Event : {alpha}")

                # Update upper bound
                upper_bound = self.estimate_upper_bound((x, alpha), v)
                tau_star = np.random.exponential(1/upper_bound) #Propose switching time
                tau_opt = tau_star
                tau_ref = np.random.exponential(self.lr) #Obtain refresh time
                tau_gibbs = np.random.exponential(self.gibbs_ref) #Obtain gibbs time


        if(self.no_exceeded_bounds > 0):
            print("No of exceeded bounds: ", self.no_exceeded_bounds)

class boomerang_sampler: 
    """
    In this code we automatically compute upper bound using arXiv:2206.11410

    Parameters
    ----------
    sigma_ref : np.array    

    mu_ref : np.array

    gradient : function

    gibbs_sampler : function

    initial_pos : tuple of np.array that contains the non gibbs and gibbs variables

    niter : int

    lr : float
    """
    def __init__(self, sigma_ref, mu_ref, gradient, initial_pos, niter, lr, initial_t_max_guess, seed=0, noisy_gradient=False, q=0.9, jax_opt=False):
        np.random.seed(seed)
        self.d = len(initial_pos)
        self.gradient = gradient
        self.initial_pos = initial_pos
        self.niter = niter
        self.lr = lr
        self.sigma_ref = sigma_ref
        self.mu_ref = mu_ref
        self.t_max = initial_t_max_guess
        self.no_refresh = 0
        self.no_switch = 0
        self.noisy_gradient = noisy_gradient
        self.q = q
        self.no_exceeded_bounds = 0
        self.no_refresh_events = 0
        self.no_horizon_events = 0
        self.jax_opt = jax_opt

    def elliptic_dynamics(self, x, v, t):
        x_t = self.mu_ref + (x - self.mu_ref) * np.cos(t) + v * np.sin(t)
        v_t = - (x - self.mu_ref) * np.sin(t) + v * np.cos(t)
        return np.array([x_t, v_t])

    def rate(self, time, pos, vel, seed=None):
        skel = self.elliptic_dynamics(pos, vel, time)
        if seed:
            grad = self.gradient(skel[0], seed)
            return (np.dot(skel[1], grad)>0)*np.dot(skel[1], grad)
        else:
            grad = self.gradient(skel[0])
            return (np.dot(skel[1], grad)>0)*np.dot(skel[1], grad)
           #return (np.dot(skel[1], self.gradient(skel[0]))>0)*np.dot(skel[1], self.gradient(skel[0]))

    def trajectory_sample(self, no_samples=None):
        if no_samples is None:
            no_samples = self.niter
        skel_t = np.vstack([x[0] for x in self.skeleton]).reshape(-1)
        skel_v = np.vstack([x[2] for x in self.skeleton])
        skel_x = np.vstack([x[1] for x in self.skeleton])

        traj_corr_samples = []
        sample_time = np.linspace(0, skel_t[-1] - 10e-5, no_samples)
        for iter_time in sample_time:
            iter_index = bisect.bisect_left(skel_t, iter_time) - 1 #t in [t_k, tK+1], index of t_k
            traj_corr_samples.append(self.elliptic_dynamics(skel_x[iter_index,:], skel_v[iter_index,:], iter_time - skel_t[iter_index])[0])

        return np.vstack(traj_corr_samples)
    
    def estimate_upper_bound(self, x, v):
        """ Estimate upper bound using Brent's method"""
        if self.noisy_gradient:
            rates = []
            for seed in range(10):
                negglobrate = lambda t: - self.rate(t, x, v, seed=seed)
                brent_min, funeval = mod_brent_corbella(self.t_max, negglobrate, eps=1e-6, round=False)
                #self.tot_fun_eval = self.tot_fun_eval + funeval
                upper_bound = - brent_min
                rates.append(upper_bound)
            #for seed i, upper bound 
            rates = np.vstack(rates)

            c, loc, scale = genextreme.fit(rates)
            upper_bound = genextreme.ppf(self.q, c, loc,scale)
            if upper_bound > 10000 or upper_bound < 0.01:
                print("WARNING: upper bound is very large/small")
            print(f"estimated upper bound: {upper_bound}")
        else:
            negglobrate = lambda t: - self.rate(t, x, v)
            brent_min, funeval = mod_brent_corbella(self.t_max, negglobrate, eps=1e-6, round=False, jax=self.jax_opt)
            #self.tot_fun_eval = self.tot_fun_eval + funeval
            upper_bound = - brent_min

        if upper_bound==0:
            print('bound became zero, adding a small perturbation')
            upper_bound = upper_bound + 10e-7
        print(f"upper bound is {upper_bound}")
        return upper_bound
    
    def sample(self):
        self.skeleton = [(0, self.initial_pos, np.random.multivariate_normal(np.zeros(self.d), np.eye(self.d), size=1).reshape(-1))] # [(time, position, velocity)] list of tuples 
        self.skel_count = 1
        T, x, v = self.skeleton[0] 
        upper_bound = self.estimate_upper_bound(x, v)
        tau_star = np.random.exponential(1/upper_bound) #Propose switching time
        tau_opt = tau_star #Tau optimal is the time since last upper bound optimisation
        tau_ref = np.random.exponential(self.lr) #Obtain refresh time

        while self.skel_count < self.niter:
            #Main loop
            u = 0 
            rejection_counter = 0
            while tau_opt == np.min(np.array([self.t_max, tau_opt, tau_ref])) and u == 0:
                # Loop invariant: tau_opt is the minimum of the three
                lambda_opt = self.rate(tau_opt, x, v)
                print(f"lambda_opt: {lambda_opt/upper_bound}")
                if np.isnan(np.min(lambda_opt)):
                    print("lambda_opt is nan")

                if(lambda_opt/upper_bound > 1):
                    self.no_exceeded_bounds += 1
                    print("No of exceeded bounds: ", self.no_exceeded_bounds / self.skel_count)
                    upper_bound = lambda_opt

                u = np.random.binomial(1, lambda_opt/upper_bound)#np.clip(lambda_opt/upper_bound, 0, 1))
                if u == 1:
                    # Accept the proposed switching time with probability lambda_opt/upper_bound
                    # Update T,x,v
                    T = T + tau_opt
                    x, v = self.elliptic_dynamics(x, v, tau_opt)
                    grad_x = self.gradient(x)

                    sigma_sqrt = np.linalg.cholesky(self.sigma_ref)
                    switch_rate = np.dot(v, grad_x)
                    skewed_grad = np.dot(sigma_sqrt, grad_x)

                    v = v - 2 * (switch_rate / np.dot(skewed_grad, skewed_grad)) * sigma_sqrt @ skewed_grad

                    self.skeleton.append((T, x, v)) #Update skeleton
                    self.skel_count += 1
                    print(f"Switch Event :{self.skel_count} at time: {T}")
                    print(x)
                    print(f"Rejection Counter :{rejection_counter}")

                    # Update upper bound
                    upper_bound = self.estimate_upper_bound(x, v)
                    tau_star = np.random.exponential(1/upper_bound) #Propose switching time
                    tau_opt = tau_star
                    tau_ref = np.random.exponential(self.lr) #Obtain refresh time
                else:
                    # Reject the proposed switching time
                    tau_star = np.random.exponential(1/upper_bound)
                    tau_opt = tau_opt + tau_star
                    rejection_counter += 1

            #Checking u=0 to enforce that if skeleton point was accepted in switch event, wait until next loop 
            if tau_ref == np.min(np.array([self.t_max, tau_opt, tau_ref])) and u == 0:
                #Refreshment time is reached
                self.no_refresh_events += 1
                T = T + tau_ref
                x, v = self.elliptic_dynamics(x, v, tau_ref)[0], np.random.multivariate_normal(np.zeros(self.d), self.sigma_ref)
                self.skeleton.append((T, x, v)) #Update skeleton
                self.skel_count += 1
                print(self.skel_count)
                print(f"Refresh Event : at time: {T}")

                # Update upper bound
                upper_bound = self.estimate_upper_bound(x, v)
                tau_star = np.random.exponential(1/upper_bound) #Propose switching time
                tau_opt = tau_star
                tau_ref = np.random.exponential(self.lr) #Obtain refresh time

            if self.t_max == np.min(np.array([self.t_max, tau_opt, tau_ref])) and u == 0:
                # Horizon is reached
                self.no_horizon_events += 1
                T = T + self.t_max
                x, v = self.elliptic_dynamics(x, v, self.t_max)

                # Update upper bound
                upper_bound = self.estimate_upper_bound(x, v)
                tau_star = np.random.exponential(1/upper_bound) #Propose switching time
                tau_opt = tau_star
                tau_ref = np.random.exponential(self.lr) #Obtain refresh time

        if(self.no_exceeded_bounds > 0):
            print("No of exceeded bounds: ", self.no_exceeded_bounds)