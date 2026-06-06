# Bench — Scripturi de colectare a măsurătorilor pentru lucrarea de licență

Aceste scripturi colectează valorile pentru tabelele `tab:cap6-env` și
`tab:cap6-perf` din capitolul 6 al lucrării. Nu modifică codul de producție;
sunt instrumente externe care interoghează sistemul (procese, Redis).

## Conținut

| Script | Scop | Tabel țintă |
|--------|------|------------|
| `env_collect.sh` | Distribuție Linux, kernel, CPU, RAM, Tracee, Python, Redis | `tab:cap6-env` |
| `perf_collect.sh` | CPU% (idle, sub încărcare), RAM (RSS MB), throughput Redis | `tab:cap6-perf` |

## Cerințe

- **Linux** (Ubuntu/Debian recomandat; `env_collect.sh` are fallback pentru macOS, dar `perf_collect.sh` are nevoie de `pidstat` Linux)
- `pidstat` — `sudo apt install sysstat`
- `redis-cli` — `sudo apt install redis-tools`
- `docker` — pentru identificarea versiunii Tracee
- `python3` — pentru afișarea versiunii

## Pas cu pas — completarea tabelelor

### Tabelul `tab:cap6-env`

Pe **gazda monitorizată**:
```bash
./bench/env_collect.sh --latex
```
Copiază rândul „Gazda monitorizată" în `Scris/capitol6_evaluare.tex`.

Pe **serverul central** (dacă diferă):
```bash
./bench/env_collect.sh --latex
```
Copiază rândul „Serverul central".

### Tabelul `tab:cap6-perf`

1. **Pornește sistemul**: `./start_server.sh` pe serverul central, `./start_agent.sh` pe gazda monitorizată.

2. **Pe gazdă, măsurarea agentului (idle + sub încărcare)**:
   ```bash
   ./bench/perf_collect.sh agent --load --duration 60
   ```
   Output:
   - `CPU agent (idle)` → rândul 1 din tab:cap6-perf
   - `CPU agent (sub încărcare)` → rândul 2
   - `RAM agent` → rândul 3

3. **Pe serverul central, măsurarea serverului + throughput**:
   ```bash
   ./bench/perf_collect.sh server --duration 60
   ```
   Output:
   - `RAM server central` → rândul 4
   - `Throughput susținut server` → rândul 5

## Note despre măsurători

- **Faza idle** = 60s fără activitate suplimentară pe gazdă. Asigură-te că nu rulează alte sarcini care consumă CPU.
- **Faza sub încărcare (workload)** generează syscall-uri sintetice prin două bucle paralele care apelează `stat`, `ls`, `cat`, `id`, `uname`. Workload-ul este deliberat ușor (~mii de syscall-uri/s) pentru a stresa agentul, nu CPU-ul gazdei.
- **Throughput** = numărul de evenimente publicate de agent pe stream-ul `events.{hostname}` în 60s, împărțit la 60. Presupune că agentul rulează activ (nu idle).
- Pentru consistență, rulează fiecare măsurătoare de 3 ori și raportează mediana. Dacă diferențele sunt mari, descrie în text scenariul exact (ce rulează pe sistem).

## Verdictul pentru rândul CPU (coloana „Verdict" din tabel)

Ținta este NF1: `< 2,5% CPU pe gazda monitorizată`. După măsurătoare:
- Dacă `CPU agent (idle)` și `CPU agent (sub încărcare)` sunt amândouă sub 2,5%, scrie `Atins` în coloana verdict.
- Altfel scrie valoarea efectivă, e.g., `2,8% (sub țintă, peste NF1)`.
