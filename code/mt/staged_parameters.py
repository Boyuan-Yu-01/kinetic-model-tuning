"""Shared-anchor A/Ea/b coordinates for staged sensitivity optimization."""
from copy import deepcopy
import math
import cantera as ct
import numpy as np
from .ae_parameters import AEParameters
from .uncertainty import _entries


class StagedParameters(AEParameters):
    """Release A, then Ea, then b without changing nominal uncertainty anchors.

    PLOG nodes have separate magnitudes and, by default, common Ea/b shifts.
    plog_shape_free releases separate Ea/b coordinates at each pressure.
    """
    physical_fields = ('A', 'Ea', 'b')

    def __init__(self, *args, stage='A', center=None, stage_a_ratio=(.5, 2.),
                 stage_ea_delta=60000., b_delta=.2, orders=None, order_anchor=None, rate_bounds=None, **kwargs):
        from .order_tuning import order_specs
        self.orders = order_specs(orders)
        self.order_anchor = order_anchor
        if self.orders:
            if order_anchor is None:
                raise ValueError('Reference order_anchor must be prepared before building order coordinates.')
            kwargs['pivot_temperature'] = float(order_anchor['temperature_K'])
        super().__init__(*args, **kwargs)
        if stage not in ('A', 'AE', 'AEB'):
            raise ValueError('stage must be A, AE, or AEB.')
        if not np.isfinite(b_delta) or b_delta <= 0:
            raise ValueError('b_delta must be positive.')
        self.b_scale = float(b_delta)
        offset = len(self.x0)
        self.b_index = {n:offset+i for i,n in enumerate(self.shape_keys)}
        self.x0 = np.r_[self.x0, np.zeros(self.ne)]
        self.source_coordinates = np.r_[self.source_coordinates, np.zeros(self.ne)]
        self.reaction_numbers += tuple(k[0] if isinstance(k,tuple) else k for k in self.shape_keys)
        self.a_map = np.pad(self.a_map, ((0,0),(0,self.ne)))
        for i,n in enumerate(self.base.reaction_numbers):
            self.a_map[i,self.b_index[self.shape_for_a[i]]] = -self.b_scale*math.log(self.pivot)
        self.coordinate_labels += [dict(reaction=k[0] if isinstance(k,tuple) else k, pressure_pa=k[1] if isinstance(k,tuple) else None, parameter='scaled_b',
            label=f'R{k} Δb/{self.b_scale:g} at fixed k(Tref)',scale=self.b_scale) for k in self.shape_keys]
        self.order_indices = {}
        self.order_origins = {}
        self.order_scales = {}
        concentration_unit = self.source_units.convert_to(1., 'kmol/m^3')
        for number, spec in self.orders.items():
            if number not in self.numbers:
                raise ValueError(f'R{number}: order tuning requires an adjustable reaction.')
            if np.isfinite(self.base.lower[self.base.reaction_numbers.index(number)]):
                raise ValueError('Published rate uncertainty factors cannot constrain changing orders; declare the reaction artificial.')
            reaction = self.base._source_gas.reaction(number-1)
            if reaction.reversible or type(reaction.rate) not in (ct.ArrheniusRate, ct.PlogRate):
                raise ValueError('Order tuning supports irreversible Arrhenius and PLOG reactions.')
            for species, bounds in spec['bounds'].items():
                if species not in reaction.orders or species not in reaction.reactants:
                    raise ValueError(f'R{number}: tunable reactant orders must already be explicit in the YAML.')
                key = (number, species)
                j = len(self.x0)
                self.order_indices[key] = j
                self.order_origins[key] = float(reaction.orders[species])
                self.order_scales[key] = float(spec['scale'])
                concentration = float(order_anchor['concentrations_kmol_m3'][species]) / concentration_unit
                if not np.isfinite(concentration) or concentration <= 0:
                    raise ValueError('Reference reactant concentrations must be positive.')
                self.x0 = np.r_[self.x0, 0.]
                self.source_coordinates = np.r_[self.source_coordinates, 0.]
                self.reaction_numbers += (number,)
                self.a_map = np.pad(self.a_map, ((0,0),(0,1)))
                for i, n in enumerate(self.base.reaction_numbers):
                    if n == number:
                        self.a_map[i,j] = -spec['scale'] * math.log(concentration)
                self.coordinate_labels.append(dict(reaction=number, parameter='scaled_order', species=species,
                    scale=spec['scale'], label=f'R{number} Δorder({species})/{spec["scale"]:g} at fixed reference rate'))
        origin = self.x0.copy() if center is None else np.asarray(center,dtype=float)
        if origin.shape != self.x0.shape or not np.isfinite(origin).all():
            raise ValueError('Invalid stage-entry coordinates.')
        self.x0 = origin.copy()
        ratio = np.asarray(stage_a_ratio,dtype=float)
        if ratio.shape!=(2,) or not 0 < ratio[0] < 1 < ratio[1]:
            raise ValueError('stage_a_ratio must bracket 1 with positive values.')
        self.a_low = np.maximum(self.a_low,self.a_map@origin+np.log(ratio[0]))
        self.a_high = np.minimum(self.a_high,self.a_map@origin+np.log(ratio[1]))
        self.lower = np.r_[np.full(self.na,-np.inf),-np.ones(self.ne),-np.ones(self.ne)]
        self.upper = np.r_[np.full(self.na,np.inf), np.ones(self.ne), np.ones(self.ne)]
        for key, j in self.order_indices.items():
            n, species = key
            low, high = self.orders[n]['bounds'][species]
            initial = self.order_origins[key]
            self.lower = np.r_[self.lower, (low-initial)/self.order_scales[key]]
            self.upper = np.r_[self.upper, (high-initial)/self.order_scales[key]]
        es = slice(self.na,self.na+self.ne);bs=slice(self.na+self.ne,self.na+2*self.ne)
        if stage=='A':
            self.lower[es]=self.upper[es]=origin[es]
        else:
            delta=float(stage_ea_delta)/self.energy_yaml
            if not np.isfinite(delta) or delta<=0: raise ValueError('stage_ea_delta must be positive.')
            self.lower[es]=np.maximum(-1,origin[es]-delta)
            self.upper[es]=np.minimum(1,origin[es]+delta)
        if stage!='AEB':self.lower[bs]=self.upper[bs]=origin[bs]
        for number, bounds in (rate_bounds or {}).items():
            number=int(number)
            if number not in self.numbers or set(bounds)-{'Ea','b'}:
                raise ValueError('rate_bounds accepts absolute Ea and b bounds for adjustable reactions.')
            for i,n in enumerate(self.base.reaction_numbers):
                if n!=number:continue
                for field in bounds:
                    lo,hi=map(float,bounds[field])
                    if not np.isfinite([lo,hi]).all() or lo>=hi:raise ValueError('Invalid physical rate bounds.')
                    j=(self.ea_index if field=='Ea' else self.b_index)[self.shape_for_a[i]]
                    scale=self.energy_yaml if field=='Ea' else self.b_scale
                    for entry in self.base._entry_indices[i]:
                        initial=float(_entries(self.base._source['reactions'][n-1])[entry][field])
                        self.lower[j]=max(self.lower[j],(lo-initial)/scale)
                        self.upper[j]=min(self.upper[j],(hi-initial)/scale)
        # Enclosing pivot boxes; physical A bounds remain coupled constraints.
        for i,n in enumerate(self.base.reaction_numbers):
            js=np.flatnonzero(self.a_map[i,self.na:])+self.na
            lo=sum(min(self.a_map[i,j]*self.lower[j],self.a_map[i,j]*self.upper[j]) for j in js)
            hi=sum(max(self.a_map[i,j]*self.lower[j],self.a_map[i,j]*self.upper[j]) for j in js)
            self.lower[i]=self.a_low[i]-hi;self.upper[i]=self.a_high[i]-lo
        # Enclosing boxes must retain a previously accepted boundary point.
        # Coupled physical constraints use 1e-12 feasibility slack; re-forming
        # the pivot box must not reject that same point at machine precision.
        near_low=(origin<self.lower)&(origin>=self.lower-1e-12)
        near_high=(origin>self.upper)&(origin<=self.upper+1e-12)
        self.lower[near_low]=origin[near_low]
        self.upper[near_high]=origin[near_high]
        self.parameter_scales=np.array([v for r in self.base._source['reactions'] for e in _entries(r)
                                       for v in (np.nan,self.energy_yaml,self.b_scale)])
        self.parameter_scales = np.r_[self.parameter_scales, np.ones(len(self.order_indices))]
        self.validate(self.x0)
        if not self._orders_feasible(self.x0):
            raise ValueError('Initial orders violate the selected sum bounds.')

    def scheme(self,x):
        """Export A/Ea using original units and apply the selected b shifts."""
        data=super().scheme(x)
        for i,n in enumerate(self.base.reaction_numbers):
            for j in self.base._entry_indices[i]:
                e=_entries(data['reactions'][n-1])[j]
                e['b']=float(e['b'])+float(self.b_scale*x[self.b_index[self.shape_for_a[i]]])
        for (number, species), j in self.order_indices.items():
            reaction = data['reactions'][number-1]
            reaction['orders'][species] = float(self.order_origins[(number,species)] + self.order_scales[(number,species)]*x[j])
            if any(v < 0 for v in reaction['orders'].values()):
                reaction['negative-orders'] = True
        return data

    def apply(self,gas,x):
        """Restore immutable SI rates before applying A, Ea and b changes."""
        if self.order_indices:
            # Changing orders changes A's dimensional conversion and Cantera's
            # immutable kinetic structure. Rebuild from YAML, retaining state.
            updated = self.make_gas(x)
            updated.state = gas.state
            return updated
        super().apply(gas,x)
        for n in self.numbers:
            reaction=gas.reaction(n-1)
            from .ae_parameters import _plain
            data=_plain(reaction.input_data)
            for i,number in enumerate(self.base.reaction_numbers):
                if number==n:
                    for j in self.base._entry_indices[i]:
                        _entries(data)[j]['b']+=self.b_scale*x[self.b_index[self.shape_for_a[i]]]
            gas.modify_reaction(n-1,ct.Reaction.from_dict(data,gas))
        return gas

    def _rate_row(self,i,n,t):
        row=self.a_map[i].copy()
        row[self.ea_index[self.shape_for_a[i]]]-=self.energy_si/(ct.gas_constant*t)
        row[self.b_index[self.shape_for_a[i]]]+=self.b_scale*math.log(t)
        return row

    def linear_monotonicity(self,t_min,t_max,direction='either'):
        """Pressure and rate-envelope proposal cuts.

        The exact temperature envelope, including interior extrema, is checked
        separately before any step can be accepted.
        """
        branches=self._pressure_branches(t_min,t_max,direction)
        rows=[*self.a_map,*(-self.a_map)];bounds=[*self.a_low,*(-self.a_high)]
        for i,n in enumerate(self.base.reaction_numbers):
            if not np.isfinite(self.base.lower[i]):continue
            # With b fixed at its source value, log(k/k0) is affine in
            # 1/T. Interior grid cuts are redundant and destabilize SLSQP.
            j=self.b_index[self.shape_for_a[i]]
            ts=([t_min,t_max] if self.lower[j]==self.upper[j]==0.
                else list(np.geomspace(t_min,t_max,33)))
            db=self.b_scale*self.x0[self.b_index[self.shape_for_a[i]]];de=self.energy_si*self.x0[self.ea_index[self.shape_for_a[i]]]
            if db and t_min < -de/(ct.gas_constant*db) < t_max:ts.append(-de/(ct.gas_constant*db))
            for t in ts:
                row=self._rate_row(i,n,t);rows.extend((row,-row));bounds.extend((self.base.lower[i],-self.base.upper[i]))
        for number, spec in self.orders.items():
            if spec.get('sum_bounds') is None:
                continue
            row = np.zeros(len(self.x0))
            origin = 0.
            for species in spec['bounds']:
                key = (number,species)
                row[self.order_indices[key]] = self.order_scales[key]
                origin += self.order_origins[key]
            low, high = spec['sum_bounds']
            rows.extend((row,-row)); bounds.extend((low-origin,origin-high))
        return [(np.vstack([c,rows]),np.r_[d,bounds]) for c,d in branches]

    def rate_envelope(self,x,t_min,t_max):
        """Check endpoints and the analytical interior extremum of log(k/k0)."""
        x=self.validate(x);a=self.a_map@x
        valid=bool(np.all(a>=self.a_low-1e-12) and np.all(a<=self.a_high+1e-12))
        for i,n in enumerate(self.base.reaction_numbers):
            db=self.b_scale*x[self.b_index[self.shape_for_a[i]]];de=self.energy_si*x[self.ea_index[self.shape_for_a[i]]]
            ts=[t_min,t_max]
            if db and t_min < -de/(ct.gas_constant*db) < t_max:ts.append(-de/(ct.gas_constant*db))
            for t in ts:
                v=self._rate_row(i,n,t)@x
                valid &= bool(self.base.lower[i]-1e-12<=v<=self.base.upper[i]+1e-12)
        return dict(valid=valid and self._orders_feasible(x),temperature_interval=[t_min,t_max],method='endpoints_and_analytic_extremum')

    def reported_multipliers(self,x):
        return [*np.exp(self.a_map@x-self.base.source_coordinates),*([None]*(2*self.ne+len(self.order_indices)))]

    def _orders_feasible(self,x):
        """Enforce sums of the selected actual orders, including negative orders."""
        for number,spec in self.orders.items():
            if spec.get('sum_bounds') is None:
                continue
            value=sum(self.order_origins[(number,s)]+self.order_scales[(number,s)]*x[self.order_indices[(number,s)]]
                      for s in spec['bounds'])
            if not spec['sum_bounds'][0]-1e-12 <= value <= spec['sum_bounds'][1]+1e-12:
                return False
        return True

    def physical_parameters(self,x):
        """Include actual orders in the all-parameter convergence audit."""
        return np.r_[super().physical_parameters(x),
                     [self.order_origins[k]+self.order_scales[k]*x[j] for k,j in self.order_indices.items()]]

    def sensitivity_steps(self,a=.02,ea=60.,b=.005):
        """Finite-difference increments, separate from the optimizer trust radius."""
        return [a]*self.na+[ea/self.energy_yaml]*self.ne+[b/self.b_scale]*self.ne+[
            self.orders[n]['step']/self.order_scales[(n,s)] for n,s in self.order_indices]


    def trust_step_scales(self):
        """Conservative joint steps; b/Ea and orders have smaller physical moves."""
        if not self.orders:
            return np.ones(len(self.x0))
        # 5 kJ/mol is an internal step scale, converted from the YAML-aware SI scale.
        return np.r_[np.ones(self.na), np.full(self.ne,min(1.,5e6/self.energy_si)),
                     np.full(self.ne,min(1.,.1/self.b_scale)),
                     [.1/self.order_scales[k] for k in self.order_indices]]

    def encode(self, model):
        """Import a saved candidate without rebasing the source bounds or priors."""
        from .uncertainty import _load, _identity
        data=_load(model)
        source=self.base._source
        if len(data['reactions'])!=len(source['reactions']) or data.get('units',{})!=source.get('units',{}):
            raise ValueError('Initial model must retain the source reaction count and YAML units.')
        for old,new in zip(source['reactions'],data['reactions']):
            if _identity(old)!=_identity(new) or len(_entries(old))!=len(_entries(new)):
                raise ValueError('Initial model reaction structure changed.')
        x=self.source_coordinates.copy()
        for key,j in self.order_indices.items():
            n,s=key
            x[j]=(data['reactions'][n-1]['orders'][s]-self.order_origins[key])/self.order_scales[key]
        assigned={}
        for i,n in enumerate(self.base.reaction_numbers):
            old,new=_entries(source['reactions'][n-1]),_entries(data['reactions'][n-1])
            indices=self.base._entry_indices[i]
            for field,index,scale in (('Ea',self.ea_index,self.energy_yaml),('b',self.b_index,self.b_scale)):
                values=[(new[j][field]-old[j][field])/scale for j in indices]
                j=index[self.shape_for_a[i]]
                if not np.allclose(values,values[0],rtol=1e-12,atol=1e-12) or j in assigned and not np.isclose(values[0],assigned[j]):
                    raise ValueError('Initial model does not respect shared shape coordinates.')
                x[j]=assigned[j]=values[0]
            values=[math.log(new[j]['A']/a) for j,a in zip(indices,self.base._origins[i])]
            if not np.allclose(values,values[0],rtol=1e-12,atol=1e-12):
                raise ValueError('Initial model does not respect shared A coordinates.')
            x[i]=values[0]-(self.a_map[i]@x-x[i])
        self.validate(x)
        if not self._orders_feasible(x):raise ValueError('Initial model violates order-sum bounds.')
        # Verify no unselected order, pressure entry, stoichiometry or rate was silently lost.
        a,b=self.make_gas(x),ct.Solution(yaml=__import__('yaml').safe_dump(data))
        for n in range(a.n_reactions):
            if a.reaction(n).orders != b.reaction(n).orders:
                if any(abs(a.reaction(n).orders.get(s,0)-v)>1e-12 for s,v in b.reaction(n).orders.items()):
                    raise ValueError('Initial model changes fixed orders.')
        for t in (600.,1800.,3000.):
            a.TPX=b.TPX=t,1e5,np.ones(a.n_species)
            if not np.allclose(a.forward_rates_of_progress,b.forward_rates_of_progress,rtol=1e-10,atol=0.):
                raise ValueError('Initial model changes fixed kinetics or species.')
        return x
