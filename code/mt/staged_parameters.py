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
                 stage_ea_delta=60000., b_delta=.2, **kwargs):
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
        es = slice(self.na,self.na+self.ne);bs=slice(self.na+self.ne,None)
        if stage=='A':
            self.lower[es]=self.upper[es]=origin[es]
        else:
            delta=float(stage_ea_delta)/self.energy_yaml
            if not np.isfinite(delta) or delta<=0: raise ValueError('stage_ea_delta must be positive.')
            self.lower[es]=np.maximum(-1,origin[es]-delta)
            self.upper[es]=np.minimum(1,origin[es]+delta)
        if stage!='AEB':self.lower[bs]=self.upper[bs]=origin[bs]
        # Enclosing pivot boxes; physical A bounds remain coupled constraints.
        for i,n in enumerate(self.base.reaction_numbers):
            js=[self.ea_index[self.shape_for_a[i]],self.b_index[self.shape_for_a[i]]]
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
        self.validate(self.x0)

    def scheme(self,x):
        """Export A/Ea using original units and apply the selected b shifts."""
        data=super().scheme(x)
        for i,n in enumerate(self.base.reaction_numbers):
            for j in self.base._entry_indices[i]:
                e=_entries(data['reactions'][n-1])[j]
                e['b']=float(e['b'])+float(self.b_scale*x[self.b_index[self.shape_for_a[i]]])
        return data

    def apply(self,gas,x):
        """Restore immutable SI rates before applying A, Ea and b changes."""
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
        return dict(valid=valid,temperature_interval=[t_min,t_max],method='endpoints_and_analytic_extremum')

    def reported_multipliers(self,x):
        return [*np.exp(self.a_map@x-self.base.source_coordinates),*([None]*(2*self.ne))]
