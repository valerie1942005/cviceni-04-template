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
    Spektrální shlukování (Spectral Clustering).

    Základní myšlenka spektrálního shlukování je jiná než u k-means nebo
    DBSCAN: místo hledání shluků přímo v původním prostoru příznaků se data
    nejprve přemapují (embedding) do nového prostoru odvozeného ze spektra
    (vlastních čísel a vlastních vektorů) Laplaciánu grafu podobnosti. Teprve
    v tomto novém prostoru se provede samotné shlukování běžným algoritmem
    (zde k-means). Samotný embedding tedy ještě žádné shluky nevytváří -
    je to jen transformace prostoru, ve které jsou shluky (na rozdíl od
    původního prostoru) snadno lineárně oddělitelné.

    Postup má tři kroky, které v tomto souboru odpovídají třem metodám:
    1. _affinity_matrix - data se převedou na graf podobnosti (afinitní matici).
    2. _laplacian       - z afinitní matice se sestaví Laplacián grafu.
    3. _spectral_embedding - z Laplaciánu se vypočítá spektrální embedding.

    Poté následuje k-means na výsledném embeddingu (viz fit()).
"""

from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from src.base import Clusterer
from src.distance import Distance

class SpectralClustering(Clusterer):
    """
    Spektrální shlukování.

    Algoritmus nešhlukuje data přímo v původním prostoru příznaků, ale
    ve dvou krocích:
      a) embedding - data se přes graf podobnosti (afinitní matici) a jeho
         Laplacián přemapují do nového prostoru o rozměru n_clusters,
         odvozeného ze spektra (vlastních čísel/vektorů) Laplaciánu,
      b) shlukování - v tomto novém prostoru se teprve spustí standardní
         shlukovací algoritmus (zde k-means), který přemapovaná data
         skutečně rozdělí do shluků.

    Samotný embedding (kroky _affinity_matrix, _laplacian,
    _spectral_embedding) tedy žádné shluky nevytváří - je to pouze
    transformace prostoru. Práci "rozdělit body do shluků" odvádí až
    k-means volaný na konci fit().
    """

    def __init__(self, n_clusters: int, sigma: float, distance: "Distance") -> None:
        """
        Inicializuje spektrální shlukování.

        Parametry
        ---------
        n_clusters : int
            Počet výsledných shluků (a zároveň počet vlastních vektorů
            použitých pro embedding).
        sigma : float
            Parametr šířky Gaussova jádra použitého při výpočtu afinitní
            matice (viz _affinity_matrix).
        distance : Distance
            Injektovaná strategie pro výpočet vzdálenosti mezi dvojicí
            vzorků (viz src/distance.py). Instance se do třídy zvenčí
            předává hotová (dependency injection), třída si ji sama
            nevytváří.
        """
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.distance = distance

        self.labels_: np.ndarray | None = None
        self.embedding_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "SpectralClustering":
        """
        Natrénuje spektrální shlukování na datech x.

        Provede tři kroky embeddingu (afinitní matice -> Laplacián ->
        spektrální embedding) a na výsledném embeddingu spustí k-means,
        který teprve provede samotné rozdělení do shluků. Bez tohoto
        posledního kroku by embedding sám o sobě žádné shluky
        nevytvořil - jde jen o transformaci prostoru, ve které jsou
        shluky snadno oddělitelné.

        Parametry
        ---------
        x : np.ndarray
            Matice dat o rozměru (n_vzorků, n_příznaků).

        Návratová hodnota
        ------------------
        SpectralClustering
            Vrací self, aby bylo možné volání zřetězit (fit(x).predict()).
        """
        w = self._affinity_matrix(x)
        l = self._laplacian(w)
        embedding, eigvals = self._spectral_embedding(l, self.n_clusters)
        self.embedding_ = embedding
        self.eigenvalues_ = eigvals

        # k-means na výsledném embeddingu
        # Zde by šel váš vlastní k-means z Cvičení 03;
        # pro samostatnost repozitáře používáme knihovní.

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
        self.labels_ = kmeans.fit_predict(embedding)
        return self

    def predict(self) -> np.ndarray:
        """
        Spektrální shlukování je transduktivní.

        Výsledný embedding je vlastností celého grafu podobnosti
        (afinitní matice a jeho Laplacián se počítají ze všech vzorků
        současně) - nejde o funkci, kterou by bylo možné aplikovat na
        jediný nový bod nezávisle na ostatních. Chcete-li zařadit nová
        data, je nutné graf, Laplacián i embedding přepočítat znovu pro
        celou (rozšířenou) množinu dat. Proto zde metoda predict() na
        rozdíl např. od k-means nová data nepřijímá a smysluplnou
        predikci pro ně nelze poskytnout.

        Vyvolává
        --------
        NotImplementedError
            Vždy - jde o záměrné, trvalé omezení algoritmu, nikoli o
            nedokončený úkol.
        """
        raise NotImplementedError(
            "Spektrální shlukování je transduktivní - embedding je vlastností celého grafu "
            "podobnosti (počítá se ze všech vzorků současně), a nová data proto nelze zařadit "
            "do shluků bez přepočtu celého grafu, Laplaciánu i embeddingu."
        )

    def _affinity_matrix(self, x: np.ndarray) -> np.ndarray:
        assert isinstance(x, np.ndarray), "Parametr 'x' musí být typu np.ndarray."
        assert x.ndim == 2, "Matice 'x' musí být 2D (n_vzorků x n_příznaků)."

        n = x.shape[0]
        w = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                d = self.distance.calculate(x[i], x[j])
                w[i, j] = np.exp(-(d ** 2) / (2 * self.sigma ** 2))

        return w
        """
        Sestaví afinitní matici (matici podobnosti) o rozměru (n, n).

        Zde se data poprvé "stávají grafem": každý vzorek je uzel grafu
        a hodnota W[i, j] je váha hrany mezi uzly i a j, vyjadřující
        jejich podobnost (nikoli vzdálenost - čím blíž si vzorky jsou,
        tím vyšší je jejich váha). Podobnost se počítá z vzdálenosti
        d(i, j) = self.distance.calculate(x[i], x[j]) pomocí Gaussova
        (RKHS/RBF) jádra:

            W[i, j] = exp( -d(i, j)^2 / (2 * sigma^2) )

        Parametr self.sigma řídí "dosah" podobnosti - malé sigma dělá
        graf řídký (podobné jsou jen velmi blízké body), velké sigma
        graf naopak zahušťuje (podobné jsou si i vzdálenější body).
        Matice W je symetrická, na diagonále má hodnotu 1 (vzdálenost
        bodu od sebe sama je 0).

        Parametry
        ---------
        x : np.ndarray
            Matice dat o rozměru (n_vzorků, n_příznaků).

        Návratová hodnota
        ------------------
        np.ndarray
            Symetrická afinitní matice o rozměru (n_vzorků, n_vzorků).
        """
        # assert  Ověřte, že x je typu np.ndarray
        # assert  Ověřte počet dimenzí x (musí být 2D: n_vzorků x n_příznaků)
        raise NotImplementedError(
            "Úkol: Implementujte sestavení afinitní matice W pomocí Gaussova jádra "
            "W[i, j] = exp(-d(i, j)^2 / (2 * sigma^2)),"
            "kde d(i, j) = self.distance.calculate(x[i], x[j])."
        )

    def _laplacian(self, w: np.ndarray) -> np.ndarray:
        assert w.ndim == 2 and w.shape[0] == w.shape[1], "Matice 'w' musí být čtvercová."
        assert np.allclose(w, w.T), "Matice 'w' musí být symetrická."

        d_diag = np.sum(w, axis=1)
        d = np.diag(d_diag)
        l = d - w

        return l
        """
        Sestaví nenormalizovaný Laplacián grafu L = D - W.

        D je diagonální matice stupňů (degree matrix): na diagonále má
        součty řádků matice W (D[i, i] = sum_j W[i, j]), mimo diagonálu
        samé nuly. Laplacián L = D - W je opět symetrická matice
        o rozměru (n, n), jejíž spektrum (vlastní čísla a vektory)
        nese informaci o struktuře shluků v grafu (viz _spectral_embedding).

        Poznámka: kromě tohoto nenormalizovaného Laplaciánu existují i
        normalizované varianty (např. symetricky normalizovaný Laplacián
        L_sym = D^(-1/2) L D^(-1/2), nebo Laplacián náhodné procházky
        L_rw = D^(-1) L). Ty jsou v praxi často robustnější, ale jejich
        implementace je nad rámec tohoto cvičení - zde pracujeme pouze
        s nenormalizovanou verzí.

        Parametry
        ---------
        w : np.ndarray
            Afinitní matice o rozměru (n_vzorků, n_vzorků), viz
            _affinity_matrix.

        Návratová hodnota
        ------------------
        np.ndarray
            Nenormalizovaný Laplacián grafu L = D - W o rozměru
            (n_vzorků, n_vzorků).
        """
        # assert  Ověřte, že w je čtvercová (a symetrická) matice
        raise NotImplementedError(
            "Úkol: Implementujte sestavení Laplaciánu grafu L = D - W, kde D je diagonální "
            "matice stupňů (součty řádků W na diagonále)."
        )

    def _spectral_embedding(self, l: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert l.ndim == 2 and l.shape[0] == l.shape[1], "Matice 'l' musí být čtvercová."

        # np.linalg.eigh vrací vlastní čísla vždy seřazená vzestupně
        eigvals, eigvecs = np.linalg.eigh(l)

        # Výběr k vlastních vektorů odpovídajících k nejmenším vlastním číslům
        embedding = eigvecs[:, :k]

        return embedding, eigvals
        """
        Vypočítá spektrální embedding dat z Laplaciánu L.

        Postup: vypočítejte vlastní čísla a vlastní vektory L. Protože je
        L symetrická matice, je k tomu vhodnou (a v tomto cvičení
        požadovanou) funkcí numpy.linalg.eigh - ta na rozdíl od obecného
        numpy.linalg.eig využívá symetrie L, je numericky stabilnější a
        vrací vlastní čísla už seřazená vzestupně.

        Klíčový (a v celém cvičení nejdůležitější) krok: z takto
        získaných n vlastních čísel a jim odpovídajících vlastních
        vektorů vyberte k vlastních vektorů odpovídajících k
        NEJMENŠÍM vlastním číslům, a poskládejte je jako sloupce do
        výsledné matice embeddingu o rozměru (n, k).

        Pozor, záměna se sudou PCA: u PCA se pro embedding/redukci
        dimenze vybírají vlastní vektory odpovídající NEJVĚTŠÍM vlastním
        číslům (směry s největším rozptylem dat). Zde je tomu záměrně
        naopak - vybíráme vlastní vektory odpovídající NEJMENŠÍM
        vlastním číslům Laplaciánu (směry "nejslabších" řezů grafu,
        které nejpřirozeněji oddělují shluky). Prohození tohoto pořadí
        (výběr největších místo nejmenších) celou metodu tiše rozbije -
        výsledný embedding bude nesmyslný, ale žádná chyba se nevyvolá.

        Parametry
        ---------
        l : np.ndarray
            Laplacián grafu o rozměru (n_vzorků, n_vzorků), viz
            _laplacian.
        k : int
            Počet vlastních vektorů (odpovídajících k nejmenším
            vlastním číslům), které tvoří výsledný embedding. V tomto
            cvičení odpovídá self.n_clusters.

        Návratová hodnota
        ------------------
        Tuple[np.ndarray, np.ndarray]
            Dvojice (embedding, eigenvalues), kde:
              - embedding je matice o rozměru (n_vzorků, k) sestavená
                z k vlastních vektorů odpovídajících k nejmenším
                vlastním číslům l (jako sloupce),
              - eigenvalues je kompletní pole všech n vlastních čísel l
                seřazené vzestupně (použije se později např. pro graf
                tzv. eigengapu).
        """
        # assert  Ověřte, že l je čtvercová matice
        raise NotImplementedError(
            "Úkol: Implementujte spektrální embedding - pomocí numpy.linalg.eigh vypočítejte "
            "vlastní čísla a vektory l, vyberte k vlastních vektorů odpovídajících k nejmenším "
            "vlastním číslům a poskládejte je jako sloupce do matice embeddingu o rozměru (n, k)."
        )
