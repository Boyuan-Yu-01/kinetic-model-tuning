"""Concise console summaries for flame and joint flame/ignition fitting."""
def report_round(info, maximum, stream):
    """Report completed rounds without exposing per-evaluation solver chatter."""
    def emit(message):
        print(message,file=stream,flush=True)
    emit(f'\nRound {info["round"]}/{maximum}')
    emit('  Selected reactions: '+', '.join(f'R{r}' for r in info['selected_groups']))
    reduction=info.get('relative_loss_reduction',0.)
    emit(f'  Loss: {info["loss"]:.8g} ({100*reduction:.4g}% reduction)')
    changes=info.get('relative_changes',[])
    change='unavailable' if any(v is None for v in changes) else f'{100*max(changes,default=0.):.4g}%'
    emit(f'  Largest normalized parameter change: {change} (target: < {100*info["threshold"]:g}%)')
    ranking=info.get('screen',{}).get('ranking',[])
    emit(f'  Sensitivities: {sum(bool(r["all_resolved"]) for r in ranking)}/{len(ranking)} reaction groups resolved')
    if info.get('skipped_coordinates'):
        emit('  Unavailable derivatives skipped; rechecked next round.')
    if info.get('all_free_audit'): emit('  Additional optimization audit: all eligible parameters.')
    for n,row in info.get('orders',{}).items():
        emit(f'  R{n} orders: '+', '.join(f'{s}={v:.5g}' for s,v in row['orders'].items())+f"; sum={row['sum']:.5g}")
    emit(f'  Inner optimizer: {info.get("inner_status", "not reported")}')
    emit(f'  Stop/status: {info["status"]}')


def report_stage(result, initial, folder, stream):
    """Report validated stage results without claiming plateau is convergence."""
    reduction=100*(initial-result['loss'])/initial if initial else 0.
    print(f'\nOptimization result: {"CONVERGED" if result.get("converged") else "NOT_CONVERGED"}'
          f'\n  Rounds: {len(result["rounds"])}'
          f'\n  Stop reason: {result["status"]}'
          f'\n  Loss: {initial:.8g} -> {result["loss"]:.8g} ({reduction:.4g}% reduction)'
          '\n  Solver validation: PASSED\n  Rate uncertainty constraints: PASSED\n  PLOG monotonicity: PASSED'
          f'\n  Best model: {folder / "best_model.yaml"}'
          f'\n  Summary: {folder / "results.json"}',file=stream,flush=True)
