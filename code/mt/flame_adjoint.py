"""Map Cantera reaction-rate adjoints into the weighted residual Jacobian."""
import numpy as np

def speed_jacobian_entries(rows, cfg, space, residual_size):
    """NaN entries request finite differences, including Ea/b and free PLOG nodes."""
    result=np.full((residual_size,len(space.x0)),np.nan)
    flames=[i for i,c in enumerate(cfg['cases']) if c.get('kind')!='ignition']
    objective=cfg.get('objectives',{})
    sw=objective.get('speed_weight',1.)
    total=sw+(objective.get('temperature_weight',1.) if objective.get('temperature_scale_K') is not None else 0.)
    if objective.get('thickness_weight') is not None:total+=objective['thickness_weight']
    if objective.get('profile') is not None:total+=objective['profile'].get('weight',1.)
    factor=np.sqrt(sw/(len(flames)*total))
    if len(flames)!=len(rows):
        weights=cfg.get('branch_weights',{'flame':1.,'ignition':1.})
        factor*=np.sqrt(weights['flame']/(weights['flame']+weights['ignition']))
    for k,i in enumerate(flames):
        values=rows[i]['speed_log_rate_sensitivities']
        for j,number in enumerate(space.base.reaction_numbers):
            if number not in (space.base.plog_free or []):
                result[k,j]=factor*values[number-1]
    return result

def flame_parameter_adjoints(flame):
    """Derivatives of log speed and outlet T for elementary Arrhenius A/Ea/b."""
    import cantera as ct
    n=flame.gas.n_reactions
    rates=[flame.gas.reaction(j).rate for j in range(n)]
    if not all(type(rate) is ct.ArrheniusRate for rate in rates):
        raise ValueError('Full flame adjoints currently require elementary Arrhenius rates.')
    energy=ct.gas_constant*1800.
    def perturb(sim,i,dp):
        j,kind=divmod(i,3);rate=rates[j];reaction=sim.gas.reaction(j)
        reaction.rate=ct.ArrheniusRate(rate.pre_exponential_factor*np.exp(dp if kind==0 else 0.),rate.temperature_exponent+(dp if kind==2 else 0.),rate.activation_energy+(energy*dp if kind==1 else 0.))
        sim.gas.modify_reaction(j,reaction)
    size=sum(d.n_components*d.n_points for d in flame.domains)
    speed=np.zeros(size);speed[flame.inlet.n_components+flame.flame.component_index('velocity')]=1./flame.velocity[0]
    temp=np.zeros(size);temp[flame.inlet.n_components+(flame.flame.n_points-1)*flame.flame.n_components+flame.flame.component_index('T')]=1.
    try:
        result=np.array([flame.solve_adjoint(perturb,3*n,g,dp=1e-5) for g in (speed,temp)])
    finally:
        for j in range(n):perturb(flame,3*j,0.)
    if not np.isfinite(result).all():raise ValueError('Invalid flame parameter adjoints.')
    return result.tolist()


def full_flame_entries(rows,cfg,space,residual_size):
    """Chain physical Arrhenius derivatives into pivot coordinates and loss weights."""
    import cantera as ct
    result=speed_jacobian_entries(rows,cfg,space,residual_size)
    ids=[i for i,c in enumerate(cfg['cases']) if c.get('kind')!='ignition'];n=len(ids)
    obj=cfg['objectives'];sw=obj.get('speed_weight',1.);tw=obj.get('temperature_weight',1.)
    total=sw+(tw if obj.get('temperature_scale_K') is not None else 0.)
    if obj.get('thickness_weight') is not None:total+=obj['thickness_weight']
    if obj.get('profile') is not None:total+=obj['profile'].get('weight',1.)
    branch=1.
    if len(ids)!=len(rows):
        w=cfg.get('branch_weights',{'flame':1.,'ignition':1.});branch=np.sqrt(w['flame']/sum(w.values()))
    mapping=np.zeros((3*len(space.base._source['reactions']),len(space.x0)))
    for i,number in enumerate(space.base.reaction_numbers):
        mapping[3*(number-1)]=space.a_map[i]
        mapping[3*(number-1)+1,space.ea_index[space.shape_for_a[i]]]=space.energy_si/(ct.gas_constant*1800.)
        mapping[3*(number-1)+2,space.b_index[space.shape_for_a[i]]]=space.b_scale
    for k,i in enumerate(ids):
        if 'flame_parameter_adjoints' not in rows[i]:continue
        jac=np.asarray(rows[i]['flame_parameter_adjoints'])@mapping
        result[k]=jac[0]*branch*np.sqrt(sw/(n*total))
        if obj.get('temperature_scale_K') is not None:result[n+k]=jac[1]*branch*np.sqrt(tw/(n*total))/obj['temperature_scale_K']
    # Arrhenius adjoints omit the explicit concentration-order derivative.
    # Those columns must use complete finite-difference residuals.
    for j in getattr(space,'order_indices',{}).values():result[:,j]=np.nan
    return result
