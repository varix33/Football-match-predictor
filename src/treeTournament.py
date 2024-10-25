import threading
import tkinter as tk
from tkinter import filedialog

try :
    from tournament import demarrerTournois
except ImportError:
    from .tournament import demarrerTournois


class ArbreTournoi:

    def __init__(self, master, model, build_match, rankingData, reraBase, scaler):
        self.master = master
        self.width = 1200
        self.height = 800
        self.nodes = []

        self.model = model
        self.build_match = build_match
        self.rankingData = rankingData
        self.reraBase = reraBase
        self.scaler = scaler

        self.master.title(f"Arbre de Tournoi de Football")
        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

        # Ajout d'un bouton en bas de la page
        self.bouton_quitter = tk.Button(self.master, text="Select file", command=self.open_csv_file, width=20, height=4)
        self.bouton_quitter.pack(side=tk.BOTTOM)

    def add_teams(self, list_results):
        i = 0
        for result in list_results:
            x = self.nodes[i]['x']
            y = self.nodes[i]['y']
            self.canvas.create_text(x, y, text=result[0])
            self.canvas.create_text(x, y+20, text=result[1])
            i += 1

    def add_nodes(self, nb_team):
        levels = int(nb_team.bit_length())  # Calcul du nombre de niveaux nécessaires

        y = self.height - 50
        y_step = (self.height - 100) // (levels - 1)
        for level in range(1, levels + 1):
            num_teams = 2 ** (levels - level)
            x_step = self.width // num_teams
            for i in range(num_teams):
                x = i * x_step + x_step / 2
                self.nodes.append({'x':x, 'y':y})
            y -= y_step

    def dessiner_arbre(self, nb_team):
        levels = int(nb_team.bit_length())  # Calcul du nombre de niveaux nécessaires

        """
        # Dessin des équipes
        y = self.height - 50
        y_step = (self.height - 100) // (levels-1)
        for level in range(1, levels+1):
            num_teams = 2 ** (levels - level)
            x_step = self.width // num_teams
            for i in range(num_teams):
                x = i * x_step + x_step / 2
                team_name = f"Équipe {i + 1}"
                self.canvas.create_text(x, y, text=team_name)
            y -= y_step"""

        # Dessin des connexions entre les équipes
        y_step = (self.height - 100) // (levels - 1)
        y1 = self.height - 50
        y2 = y1 - y_step
        for level in range(1, levels):
            num_teams = 2 ** (levels - level)
            x_step = self.width // num_teams
            for i in range(num_teams):
                x1 = i * x_step + x_step / 2
                x2 = (i -i%2 + 0.5) * x_step + x_step / 2
                self.canvas.create_line(x1, y1-15, x2, y2+30, width=2, fill="black")
            y1 -= y_step
            y2 -= y_step

    def open_csv_file(self):

        filename = filedialog.askopenfilename(
            title="Select Groups File", filetypes=[("CSV Files", "*.csv")]
        )

        if filename:
            try:
                nb_group, results = demarrerTournois(filename,self.model, self.build_match, self.rankingData, self.reraBase, self.scaler)
                nb_team = (len(results) + 1) // 2

                for i in range(nb_group//2):
                    temp = results[4 * i + 1]
                    results[4 * i + 1] = results[4 * i + 3]
                    results[4 * i + 3] = temp
                self.add_nodes(nb_team)
                self.dessiner_arbre(nb_team)
                self.add_teams(results)
            except Exception as e:
                print("Une erreur s'est produite :", e)
        else:
            print("Aucun fichier sélectionné.")



