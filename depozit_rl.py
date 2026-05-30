import random
import warnings
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Shipment:
    pr_name: str
    days_left: int
    quantity: int

class Product:
    def __init__(self, name, purchase_price, selling_price, stock, storage_cost, supplier_delay, fixed_order_cost):
        self.name = name
        self.purchase_price = purchase_price
        self.selling_price = selling_price
        self.stock = self.stoc_initial = stock
        self.storage_cost = storage_cost
        self.supplier_delay = supplier_delay
        self.fixed_order_cost = fixed_order_cost


class Warehouse:
    def __init__(self, budget, produse: Dict[str, Product], df_istoric: pd.DataFrame):
        self.buget_initial = budget
        self.budget = budget
        self.produse = produse
        self.pending_shipments: List[Shipment] = []
        self.x = {}
        self.baza_medie_cerere = {}
        self.prag_trafic_mediu = df_istoric['traffic_congestion'].mean()

        for nume_p in self.produse.keys():
            cerere_maxima = df_istoric[f'demand_{nume_p}'].max()
            self.x[nume_p] = max(3, cerere_maxima / 4)
            self.baza_medie_cerere[nume_p] = df_istoric[f'demand_{nume_p}'].mean()

    def reset(self):
        self.budget = self.buget_initial
        self.pending_shipments = []
        for p in self.produse.values():
            p.stock = p.stoc_initial
        return self.get_stare_curenta({nume: "MICA" for nume in self.produse.keys()})

    def get_stare_curenta(self, prognoze_ml: Dict[str, str]) -> tuple:
        stare_stoc = []
        stare_ml = []
        for nume_p, prod in self.produse.items():
            prag = self.x[nume_p]
            if prod.stock < prag:
                stare_stoc.append("CRITIC")
            elif prod.stock <= 2 * prag:
                stare_stoc.append("MEDIU")
            else:
                stare_stoc.append("SUFICIENT")
            stare_ml.append(prognoze_ml[nume_p])
        return (tuple(stare_stoc), tuple(stare_ml))

    def plaseaza_comanda(self, nume_produs, cantitate):
        if cantitate <= 0:
            return 0, False
        produs = self.produse[nume_produs]
        cost_total = produs.fixed_order_cost + (cantitate * produs.purchase_price)

        if self.budget >= cost_total:
            self.budget -= cost_total
            self.pending_shipments.append(Shipment(nume_produs, produs.supplier_delay, cantitate))
            return cost_total, True
        return 0, False

    def proceseaza_vanzari_zilnice(self, cereri_zi: Dict[str, int], costuri_comenzi: Dict[str, int],
                                   comenzi_lansate: Dict[str, bool], cereri_brute_weekend: Dict[str, int] = None):
        marfa_sosita = {p: 0 for p in self.produse.keys()}
        for s in self.pending_shipments:
            s.days_left -= 1
            if s.days_left == 0:
                self.produse[s.pr_name].stock += s.quantity
                marfa_sosita[s.pr_name] += s.quantity
        self.pending_shipments = [s for s in self.pending_shipments if s.days_left > 0]

        recompense_per_produs = {}
        statistici_zi = {}

        for nume_p, produs in self.produse.items():
            stoc_inainte = produs.stock
            cerere = cereri_zi[nume_p]

            vandut = min(stoc_inainte, cerere)
            pierdut = cerere - vandut
            produs.stock -= vandut

            venit = vandut * produs.selling_price
            cost_stocare = produs.stock * produs.storage_cost

            penalizare_stockout = pierdut * (produs.selling_price * 1.5)
            cost_comanda_p = costuri_comenzi[nume_p]

            bonus_disponibilitate = 0
            if produs.stock >= self.x[nume_p] and cerere > 0:
                bonus_disponibilitate = produs.selling_price * 0.2

            recompense_per_produs[
                nume_p] = venit - cost_comanda_p - cost_stocare - penalizare_stockout + bonus_disponibilitate
            self.budget += venit

            cerere_afisata = cerere
            if cereri_brute_weekend and nume_p in cereri_brute_weekend and cerere == 0:
                cerere_afisata = cereri_brute_weekend[nume_p]

            statistici_zi[nume_p] = {
                "cerut": cerere_afisata, "vandut": vandut, "pierdut": pierdut,
                "stoc_vechi": stoc_inainte, "stoc_nou": produs.stock,
                "sosita": marfa_sosita[nume_p], "comanda_lansata": comenzi_lansate[nume_p]
            }

        return recompense_per_produs, statistici_zi


def genereaza_date_proiect():
    nume_fisier_date = "date_istorice_depozit.csv"
    np.random.seed(42)
    n_ore = 1000
    timestamps = pd.date_range(start='2026-01-01', periods=n_ore, freq='h')

    df_date = pd.DataFrame({
        'day_of_week': timestamps.dayofweek,
        'hour': timestamps.hour,
        'weather_severity': np.random.uniform(0, 1, n_ore).round(2),
        'traffic_congestion': np.random.uniform(0, 10, n_ore).round(2)
    })

    ponderi = {
        "Ambreiaj": (12, 4, 1),
        "Placute_Frana": (22, 8, 2),
        "Alternator": (6, 2, 0.5),
        "Filtru_Ulei": (30, 12, 2),
        "Amortizor": (10, 3, 1)
    }

    for nume_p, (b_sin, amp_sin, mult_trafic) in ponderi.items():
        baza_cerere = b_sin + amp_sin * np.sin(2 * np.pi * df_date['hour'] / 24)
        cerere_bruta = baza_cerere + (df_date['traffic_congestion'] * mult_trafic)

        for idx in range(len(df_date)):
            if df_date.loc[idx, 'day_of_week'] in [5, 6]:
                cerere_bruta.iloc[idx] *= np.random.uniform(0.1, 0.3)

            if df_date.loc[idx, 'weather_severity'] > 0.85:
                cerere_bruta.iloc[idx] *= np.random.uniform(0.0, 0.15)

            if nume_p in ["Ambreiaj", "Alternator"] and np.random.random() < 0.25:
                cerere_bruta.iloc[idx] = 0

        df_date[f'demand_{nume_p}'] = np.clip(cerere_bruta, 0, 150).astype(int)

    df_date.to_csv(nume_fisier_date, index=False, encoding='utf-8')
    return pd.read_csv(nume_fisier_date)


df_istoric = genereaza_date_proiect()

portofoliu_produse = {
    "Ambreiaj": Product("Ambreiaj", purchase_price=600, selling_price=1100, stock=12, storage_cost=0.06,
                        supplier_delay=4, fixed_order_cost=200),
    "Placute_Frana": Product("Placute_Frana", purchase_price=90, selling_price=160, stock=80, storage_cost=0.06,
                             supplier_delay=1, fixed_order_cost=50),
    "Alternator": Product("Alternator", purchase_price=450, selling_price=750, stock=15, storage_cost=0.42,
                          supplier_delay=3, fixed_order_cost=150),
    "Filtru_Ulei": Product("Filtru_Ulei", purchase_price=25, selling_price=45, stock=150, storage_cost=0.03,
                           supplier_delay=1, fixed_order_cost=45),
    "Amortizor": Product("Amortizor", purchase_price=180, selling_price=320, stock=40, storage_cost=0.2,
                         supplier_delay=3, fixed_order_cost=80)
}

depozit = Warehouse(budget=10000, produse=portofoliu_produse, df_istoric=df_istoric)

X_train = df_istoric[['day_of_week', 'hour', 'weather_severity', 'traffic_congestion']]
modele_ml = {}
for nume_p in portofoliu_produse.keys():
    y_train = df_istoric[f'demand_{nume_p}']
    modele_ml[nume_p] = DecisionTreeRegressor(max_depth=5, random_state=42)
    modele_ml[nume_p].fit(X_train, y_train)

actiuni_posibile = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50]
Q_table = {}
alpha, gamma, epsilon = 0.2, 0.9, 0.5
numar_episoade = 5000

print(f"Antrenare Agent Q-Learning ({numar_episoade} episoade)...")
comenzi_tampon_weekend = {p: 0 for p in depozit.produse.keys()}

for episod in range(numar_episoade):
    stare_curenta = depozit.reset()
    epsilon_curent = max(0.05, epsilon * (1 - episod / numar_episoade))
    comenzi_tampon_weekend = {p: 0 for p in depozit.produse.keys()}

    for _ in range(30):
        zi_saptamana = random.randint(0, 6)
        fact_zi = [zi_saptamana, random.uniform(0, 1), random.uniform(0, 10)]

        is_weekend = zi_saptamana in [5, 6]
        is_luni = zi_saptamana == 0
        is_bad_weather = fact_zi[1] > 0.85

        prognoze_zi = {}
        for p in depozit.produse.keys():
            pred = modele_ml[p].predict([[fact_zi[0], 12, fact_zi[1], fact_zi[2]]])[0]
            prognoze_zi[p] = "MARE" if pred > depozit.baza_medie_cerere[p] else "MICA"

        stare_curenta = depozit.get_stare_curenta(prognoze_zi)

        if stare_curenta not in Q_table:
            Q_table[stare_curenta] = {p: [0] * len(actiuni_posibile) for p in depozit.produse.keys()}

        actiuni_alese = {}
        cost_orders_zi = {p: 0 for p in depozit.produse.keys()}
        comenzi_lansate_zi = {p: False for p in depozit.produse.keys()}

        for p in depozit.produse.keys():
            if random.random() < epsilon_curent:
                idx_act = random.randint(0, len(actiuni_posibile) - 1)
            else:
                idx_act = np.argmax(Q_table[stare_curenta][p])

            cantitate_cmd = actiuni_posibile[idx_act]
            actiuni_alese[p] = idx_act

            cost_cmd, succes = depozit.plaseaza_comanda(p, cantitate_cmd)
            cost_orders_zi[p] = cost_cmd
            comenzi_lansate_zi[p] = succes

        cereri_generate_zi = {}
        for p in depozit.produse.keys():
            baza = depozit.baza_medie_cerere[p]
            val = max(0, int(baza + random.randint(-4, 6)))
            if is_bad_weather:
                val = int(val * 0.1)
            if p in ["Ambreiaj", "Alternator"] and random.random() < 0.25:
                val = 0
            cereri_generate_zi[p] = val

        cerere_efectiva_rampa = {}
        for p in depozit.produse.keys():
            if is_weekend:
                comenzi_tampon_weekend[p] += cereri_generate_zi[p]
                cerere_efectiva_rampa[p] = 0
            elif is_luni:
                cerere_efectiva_rampa[p] = cereri_generate_zi[p] + comenzi_tampon_weekend[p]
                comenzi_tampon_weekend[p] = 0
            else:
                cerere_efectiva_rampa[p] = cereri_generate_zi[p]

        recompense_p, _ = depozit.proceseaza_vanzari_zilnice(cerere_efectiva_rampa, cost_orders_zi, comenzi_lansate_zi)
        stare_urmatoare = depozit.get_stare_curenta(prognoze_zi)

        if stare_urmatoare not in Q_table:
            Q_table[stare_urmatoare] = {p: [0] * len(actiuni_posibile) for p in depozit.produse.keys()}

        for p in depozit.produse.keys():
            q_vechi = Q_table[stare_curenta][p][actiuni_alese[p]]
            max_q_viitor = np.max(Q_table[stare_urmatoare][p])
            Q_table[stare_curenta][p][actiuni_alese[p]] = q_vechi + alpha * (
                    recompense_p[p] + gamma * max_q_viitor - q_vechi)

print("\n🚀 START SIMULARE")
stare_curenta = depozit.reset()
istoric_buget = [depozit.budget]

np.random.seed(100)
zile_saptamana = [i % 7 for i in range(30)]
vreme_test = np.random.uniform(0, 1, 30).round(2)
trafic_test = np.random.uniform(0, 10, 30).round(2)

backlog_test_weekend = {p: 0 for p in depozit.produse.keys()}

for zi in range(30):
    zi_nume = ["LUNI", "MARȚI", "MIERCURI", "JOI", "VINERI", "SÂMBĂTĂ", "DUMINICĂ"][zile_saptamana[zi]]
    is_weekend_test = zile_saptamana[zi] in [5, 6]
    is_luni_test = zile_saptamana[zi] == 0
    is_vreme_rea_test = vreme_test[zi] > 0.85

    prognoze_zi = {}
    for p in depozit.produse.keys():
        pred = modele_ml[p].predict([[zile_saptamana[zi], 12, vreme_test[zi], trafic_test[zi]]])[0]
        prognoze_zi[p] = "MARE" if pred > depozit.baza_medie_cerere[p] else "MICA"

    stare_curenta = depozit.get_stare_curenta(prognoze_zi)

    cost_orders_zi = {p: 0 for p in depozit.produse.keys()}
    comenzi_lansate_zi = {p: False for p in depozit.produse.keys()}

    for p in depozit.produse.keys():
        if stare_curenta in Q_table:
            idx_act = np.argmax(Q_table[stare_curenta][p])
        else:
            idx_act = 0

        cantitate_cmd = actiuni_posibile[idx_act]
        cost_cmd, succes = depozit.plaseaza_comanda(p, cantitate_cmd)
        cost_orders_zi[p] = cost_cmd
        comenzi_lansate_zi[p] = succes

    cereri_generate_test = {}
    for p in depozit.produse.keys():
        baza = depozit.baza_medie_cerere[p]
        factor_mediu = 1.3 if trafic_test[zi] > depozit.prag_trafic_mediu else 0.7
        val = max(0, int(baza * factor_mediu + random.randint(-2, 4)))

        if is_vreme_rea_test:
            val = int(val * 0.05)
        if p in ["Ambreiaj", "Alternator"] and random.randint(1, 100) <= 25:
            val = 0
        cereri_generate_test[p] = val

    cerere_finala_test_zi = {}
    for p in depozit.produse.keys():
        if is_weekend_test:
            backlog_test_weekend[p] += cereri_generate_test[p]
            cerere_finala_test_zi[p] = 0
        elif is_luni_test:
            cerere_finala_test_zi[p] = cereri_generate_test[p] + backlog_test_weekend[p]
            backlog_test_weekend[p] = 0
        else:
            cerere_finala_test_zi[p] = cereri_generate_test[p]

    _, statistici = depozit.proceseaza_vanzari_zilnice(cerere_finala_test_zi, cost_orders_zi, comenzi_lansate_zi,
                                                       cereri_generate_test if is_weekend_test else None)
    istoric_buget.append(depozit.budget)

    if 31 > zi:
        status_zi = "☀️ NORMALĂ"
        if is_weekend_test: status_zi = "⛺ WEEKEND"
        if is_vreme_rea_test: status_zi = "🛑 VREME EXTREMĂ"

        print(
            f"\n📌 ZIUA {zi + 1} ({zi_nume}) [{status_zi}] | Buget: {depozit.budget:,.1f} Lei | Vreme: {vreme_test[zi]}")
        for p, info in statistici.items():
            cmd_text = f"🚚 Comandat +{actiuni_posibile[np.argmax(Q_table[stare_curenta][p]) if stare_curenta in Q_table else 0]}" if \
                info['comanda_lansata'] else "📦 Pas / Stoc OK"
            if info['sosita'] > 0: cmd_text += f" | 🎉 Descărcat +{info['sosita']} buc"
            print(
                f"   🔹 {p:<14} -> Cerut: {info['cerut']} | Vândut: {info['vandut']} | Pierdut: {info['pierdut']} | Stoc nou: {info['stoc_nou']} | {cmd_text}")

print(f"\n💰 SIMULARE FINALIZATĂ! Buget final depozit: {depozit.budget:,.2f} Lei")

plt.figure(figsize=(10, 5))
plt.plot(istoric_buget, marker='o', color='dodgerblue', linewidth=2, label='Evoluție Fonduri Depozit')
plt.axhline(y=10000, color='crimson', linestyle='--', label='Buget inițial (10.000 Lei)')
plt.title('Management cu Cerere Realistă', fontsize=12, fontweight='bold')
plt.xlabel('Zile din Luna de Test', fontsize=10)
plt.ylabel('Buget Disponibil', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

