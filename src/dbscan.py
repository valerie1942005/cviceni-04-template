# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Na rozdíl od k-means (Cvičení 03) DBSCAN nevyžaduje předem zadaný počet
    shluků k - ten se přirozeně "vynoří" z hustoty dat na základě dvou
    parametrů: poloměru okolí eps a minimálního počtu sousedů min_samples.
    Algoritmus navíc umí explicitně označit body, které do žádného shluku
    nepatří, jako šum (noise), s konvencí štítku -1 (stejně jako
    sklearn.cluster.DBSCAN).
"""

from typing import Optional

import numpy as np

from src.base import Clusterer
from src.distance import Distance


class DBSCAN(Clusterer):
    """
    Shlukovací algoritmus DBSCAN.

    Algoritmus rozděluje body datové sady do tří kategorií:

    - **jádrové body (core points)** - body, které mají ve svém
      eps-okolí alespoň min_samples sousedů (včetně sebe sama);
    - **hraniční body (border points)** - body, které samy nejsou
      jádrové (mají méně než min_samples sousedů), ale leží v
      eps-okolí nějakého jádrového bodu, a jsou tedy z jádrového bodu
      dosažitelné;
    - **šum (noise)** - body, které nejsou ani jádrové, ani hraniční;
      těmto bodům je přiřazen štítek -1.

    Počet výsledných shluků k **není** parametrem algoritmu - vzniká
    až za běhu jako vedlejší produkt hustoty dat a zvolených hodnot
    eps a min_samples.

    Atributy
    --------
    eps : float
        Poloměr okolí, ve kterém se vyhledávají sousedé bodu.
    min_samples : int
        Minimální počet bodů (včetně bodu samotného) v eps-okolí
        potřebný k tomu, aby byl bod považován za jádrový.
    distance : Distance
        Injektovaný objekt pro výpočet vzdálenosti/nepodobnosti mezi
        dvěma vzorky (viz src.distance.Distance). Instance třídy
        DBSCAN si žádnou konkrétní implementaci vzdálenosti sama
        nevytváří - je jí vždy dodána zvenčí (dependency injection),
        stejně jako v předchozích cvičeních.
    labels_ : Optional[np.ndarray]
        Výsledné štítky shluků získané metodou fit(). Hodnota -1
        označuje šum. Dokud fit() neproběhne, je None.
    """

    def __init__(self, eps: float, min_samples: int, distance: "Distance") -> None:
        """
        Vytvoří instanci DBSCAN s danými hyperparametry a vzdálenostní
        mírou.

        Parametry
        ---------
        eps : float
            Poloměr eps-okolí použitý při vyhledávání sousedů bodu.
        min_samples : int
            Minimální počet bodů v eps-okolí (včetně bodu samotného),
            aby byl bod považován za jádrový.
        distance : Distance
            Objekt implementující výpočet vzdálenosti mezi dvěma
            vzorky (dependency injection - instance se vytváří mimo
            třídu DBSCAN a předává se sem hotová).

        Návratová hodnota
        ------------------
        None
        """
        self.eps: float = eps
        self.min_samples: int = min_samples
        self.distance: "Distance" = distance
        self.labels_: Optional[np.ndarray] = None

    def _region_query(self, x: np.ndarray, point_idx: int, eps: float) -> list[int]:
        assert isinstance(x, np.ndarray), "Parametr 'x' musí být typu np.ndarray."
        assert x.ndim == 2, "Matice 'x' musí být 2D."
        assert 0 <= point_idx < x.shape[0], "Neplatný index 'point_idx'."

        neighbors = []
        for i in range(x.shape[0]):
            if self.distance.calculate(x[point_idx], x[i]) <= eps:
                neighbors.append(i)

        return neighbors
        """
        Najde eps-okolí bodu x[point_idx].

        Projde všechny vzorky v x a pro každý z nich spočítá vzdálenost
        k bodu x[point_idx] pomocí self.distance.calculate(...). Vrátí
        seznam indexů všech bodů, jejichž vzdálenost od x[point_idx] je
        menší nebo rovna eps. Bod samotný (point_idx) je v tomto
        seznamu vždy zahrnut, protože jeho vzdálenost sám od sebe je 0.

        Parametry
        ---------
        x : np.ndarray
            Matice dat o rozměru (n_vzorků, n_příznaků).
        point_idx : int
            Index bodu v x, jehož okolí se hledá.
        eps : float
            Poloměr okolí.

        Návratová hodnota
        ------------------
        list[int]
            Seznam indexů bodů z x ležících v eps-okolí bodu
            x[point_idx] (včetně point_idx samotného).
        """
        # assert  Ověřte, že x je typu np.ndarray a je 2D (n_vzorků x n_příznaků)
        # assert  Ověřte, že point_idx je platný index v rozsahu [0, x.shape[0])
        raise NotImplementedError(
            "Úkol: Implementujte vyhledání eps-okolí bodu x[point_idx] - pro každý bod "
            "v x spočítejte self.distance.calculate(x[point_idx], x[i]) a vraťte indexy "
            "všech bodů, jejichž vzdálenost je <= eps (včetně point_idx samotného)."
        )

    def fit(self, x: np.ndarray) -> "DBSCAN":
        assert isinstance(x, np.ndarray), "Parametr 'x' musí být typu np.ndarray."
        assert x.ndim == 2, "Matice 'x' musí být 2D (n_vzorků x n_příznaků)."

        n_samples = x.shape[0]
        # 1. Pole štítků (start: vše šum -1) a evidence navštívených bodů
        self.labels_ = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        # 2. Pro každý dosud nenavštívený bod i:
        for i in range(n_samples):
            if visited[i]:
                continue

            # a. označte i jako navštívený
            visited[i] = True

            # b. vyhledejte sousedy
            neighbors = self._region_query(x, i, self.eps)

            # c. kontrola na šum / hraniční bod
            if len(neighbors) < self.min_samples:
                # Zůstává jako šum (štítek -1), později může být zařazen jako hraniční bod
                continue

            # d. i je jádrový bod — založte nový shluk a expandujte ho
            self.labels_[i] = cluster_id

            # Inicializace fronty pro expanzi shluku
            seed_set = neighbors.copy()
            in_seed = np.zeros(n_samples, dtype=bool)
            for n in seed_set:
                in_seed[n] = True

            # Odstraníme z fronty samotný výchozí bod (už byl zpracován)
            if in_seed[i]:
                seed_set.remove(i)
                in_seed[i] = False

            # Procházení fronty k prozkoumání (BFS expanze)
            j = 0
            while j < len(seed_set):
                current_p = seed_set[j]

                if not visited[current_p]:
                    visited[current_p] = True
                    current_neighbors = self._region_query(x, current_p, self.eps)

                    # Pokud je bod v seed_set také jádrový, přidáme jeho sousedy
                    if len(current_neighbors) >= self.min_samples:
                        for n in current_neighbors:
                            if not in_seed[n]:
                                seed_set.append(n)
                                in_seed[n] = True

                # Pokud bod zatím nemá přiřazený shluk (byl původně vyhodnocen jako šum)
                if self.labels_[current_p] == -1:
                    self.labels_[current_p] = cluster_id

                j += 1

            # Expanze aktuálního shluku byla dokončena, navyšujeme ID pro další shluk
            cluster_id += 1

        # 3. Vrátíme instanci
        return self
        """
        Natrénuje DBSCAN na datech x a uloží výsledné štítky shluků do
        self.labels_.

        Klasický algoritmus (Ester et al., 1996) pracuje takto:

        1. Vytvořte pole štítků o délce n_vzorků, na začátku např.
           celé vyplněné hodnotou -1 (sentinel "zatím nenavštíveno /
           zatím bez shluku"). Zvlášť si veďte evidenci toho, které
           body už byly navštíveny (např. pomocí množiny/pole
           booleovských příznaků) - navštívení bodu a jeho konečné
           zařazení do shluku jsou dvě různé věci a je potřeba je
           rozlišovat.

        2. Pro každý dosud nenavštívený bod i:
           a. Označte bod i jako navštívený.
           b. Zavolejte self._region_query(x, i, self.eps) a získejte
              seznam sousedů (eps-okolí bodu i).
           c. Pokud má bod i méně než self.min_samples sousedů, není
              jádrový - prozatím jej označte jako šum (štítek -1).
              Pozor: toto označení je pouze prozatímní! Bod, který
              sám není jádrový, se může později stát **hraničním**
              bodem**, pokud bude objeven jako soused nějakého jiného
              jádrového bodu během expanze shluku (viz krok d) - v tom
              případě se jeho štítek přepíše na id daného shluku.
           d. Pokud má bod i alespoň self.min_samples sousedů, jde o
              **jádrový bod** - založte nový shluk (nové cluster id) a
              expandujte jej: udržujte frontu/množinu bodů k
              prozkoumání, na začátku rovnou sousedům bodu i. Pro
              každý bod z této fronty:
                 - přiřaďte mu aktuální cluster id (pokud ho ještě
                   nemá, resp. přepište jeho případné dřívější
                   označení šumem);
                 - pokud tento bod ještě nebyl navštíven, označte ho
                   jako navštívený a zjistěte jeho vlastní eps-okolí
                   (_region_query). Pokud i on má alespoň
                   self.min_samples sousedů (je také jádrový), přidejte
                   jeho dosud nezahrnuté sousedy do fronty k
                   prozkoumání (tím se shluk dále "šíří" hustotou dat -
                   klasická BFS/DFS expanze regionu).
              Tímto způsobem se do shluku postupně zahrnou jak další
              jádrové body, tak hraniční body na okraji shluku.

        3. Po zpracování všech bodů uložte výsledné štítky do
           self.labels_ a vraťte self (pro řetězení volání
           fit(X).predict()).

        Shrnutí kategorií bodů:
        - **jádrový bod (core point)** - má >= min_samples sousedů v
          eps-okolí (včetně sebe);
        - **hraniční bod (border point)** - sám má < min_samples
          sousedů, ale je součástí eps-okolí nějakého jádrového bodu,
          a je tedy tímto jádrovým bodem "dosažitelný";
        - **šum (noise)** - ani jádrový, ani hraniční bod; ve výsledných
          štítcích je označen hodnotou -1.

        Důležité: počet výsledných shluků k není vstupním parametrem
        této metody ani konstruktoru - vyplyne až za běhu algoritmu z
        hustoty dat vzhledem k eps a min_samples.

        Parametry
        ---------
        x : np.ndarray
            Matice dat o rozměru (n_vzorků, n_příznaků).

        Návratová hodnota
        ------------------
        DBSCAN
            Vrací self (self.labels_ je po volání této metody naplněno).
        """
        # assert  Ověřte, že x je typu np.ndarray
        # assert  Ověřte počet dimenzí (musí být 2D: n_vzorků x n_příznaků)
        raise NotImplementedError(
            "Úkol: Implementujte algoritmus DBSCAN podle popisu v docstringu této metody "
            "- pro každý nenavštívený bod zjistěte jeho eps-okolí přes self._region_query, "
            "rozhodněte, zda je jádrový (>= self.min_samples sousedů), a pokud ano, "
            "expandujte kolem něj nový shluk (BFS/DFS). Body, které nejsou jádrové ani "
            "nejsou dosažitelné z žádného jádrového bodu, označte jako šum (-1). Výsledek "
            "uložte do self.labels_ a vraťte self."
        )

    def predict(self) -> np.ndarray:
        """
        DBSCAN nemá metodu predict() pro nová data.

        DBSCAN je transduktivní algoritmus - nevytváří žádné centroidy
        ani jiný obecný model prostoru příznaků, pouze rozdělí do
        shluků (a šumu) přesně ta data, na kterých proběhlo fit().
        Nemá tedy smysluplný způsob, jak zařadit zcela nový, dosud
        neviděný bod do některého z nalezených shluků - takové
        zařazení by u husotně založeného algoritmu vyžadovalo znovu
        prohledat okolí nového bodu vůči celé původní datové sadě.
        Z tohoto důvodu tato metoda pro nová data nic nepredikuje a
        místo toho vždy vyvolává výjimku.

        Návratová hodnota
        ------------------
        np.ndarray
            Tato metoda nikdy nevrací hodnotu - vždy vyvolá výjimku
            NotImplementedError (viz výše).

        Vyvolává
        --------
        NotImplementedError
            Vždy - DBSCAN nemá jak predikovat na nových datech.
        """
        raise NotImplementedError(
            "DBSCAN je transduktivní algoritmus - nemá centroidy ani jiný model, který by "
            "šlo použít na nová data, a proto nepredikuje. Jediný smysluplný výstup je "
            "self.labels_ získané metodou fit() na trénovacích datech."
        )
