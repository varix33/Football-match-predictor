import csv
from itertools import combinations
import random

# Global variables :
interfaceGraphiqueElimDirect = []

"""
Documentation : 
    def write_file_csv(teams_list, file_name,nb_groups)
        Args : 
            teams_list : list of the teams to add
            file_name : name of the csv file which will be created
            nb_groups : number of groups to create
     this function create a csv file according to the parameters
"""
def write_file_csv(teams_list, file_name, nb_groups):
    nb_team = len(teams_list)
    mod = nb_team % nb_groups

    if mod != 0:
        raise ValueError("Le nombre total d'équipes n'est pas divisible par le nombre de groupes ! ")
    team_by_group = nb_team // nb_groups

    groups = []
    for i in range(nb_groups):
        group = teams_list[i * team_by_group: (i + 1) * team_by_group]
        groups.append(group)

    if mod > 0:
        for i in range(mod):
            groups[i].append(teams_list[nb_groups * team_by_group + i])

    groups = list(map(list, zip(*groups)))

    # Écriture dans le fichier CSV
    with open(file_name, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')

        csvwriter.writerow([f"Group {chr(65 + i)}" for i in range(nb_groups)])

        # Écrire les équipes par groupe
        for teams in groups:
            csvwriter.writerow(teams)


"""
Documentation : 
    def lire_fichier_csv_groupes(nom_fichier)
        Args : 
            nom_fichier : name of the file to load

         return :
             groupes : a dictionnary containing the groups and the teams in each group
    This function load an existing csv file else it return an  exception
    Verification of the number of groups : Teams are divided into n = 2^k for k ∈ N groups
"""
def lire_fichier_csv_groupes(nom_fichier):
    try:
        groupes = {}
        with open(nom_fichier, 'r', newline='', encoding='utf-8') as fichier_csv:
            lecteur_csv = csv.reader(fichier_csv, delimiter=';')
            lignes = list(lecteur_csv)

            # Récupération des entêtes de groupe
            entetes = lignes[0]
            # Initialisation des groupes dans le dictionnaire
            for entete in entetes:
                groupes[entete] = []

            # Ajout des pays dans leurs groupes respectifs
            for ligne in lignes[1:]:
                for index, pays in enumerate(ligne):
                    groupe = entetes[index]
                    groupes[groupe].append(pays)
            nombre_groupes = len(groupes)
            k = 0
            while 2 ** k < nombre_groupes:
                k += 1

            if nombre_groupes != 2 ** k:
                raise ValueError("Le nombre de groupes n'est pas égal à 2^k.")
        return groupes
    except FileNotFoundError:
        print(f"Le fichier '{nom_fichier}' est introuvable.")
        return None
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")
        return None


"""
Documentation : 
    def calculer_points_matchs_par_groupe(groupes)
        Args : 
            groupes : dictionnary of the groups and teams associated, created by the function : lire_fichier_csv_groupes(nom_fichier)

         return :
             equipes_qualifiees_par_groupe :  dictionnary containing the groups with the 2 best teams according to the points won
    Winner : Add 3 points to the winner
    Draw : 1 points for each team 
    Print the probabilitie of the winner team

"""
def calculer_points_matchs_par_groupe(groupes, model, build_match, rankingData, reraBase, scaler):
    equipes_qualifiees_par_groupe = {groupe: [] for groupe in groupes}
    for groupe, equipes in groupes.items():
        points = {equipe: 0 for equipe in equipes}
        matchs = list(combinations(equipes, 2))

        for equipe1, equipe2 in matchs:
            print(f"Match: {equipe1} contre {equipe2}")

            match = build_match(rankingData, reraBase, equipe1, equipe2, "FIFA World Cup", True, scaler)
            probabilities = model.predict_proba(match)
            proba_equipe1_gagne = probabilities[0][0]
            proba_equipe2_gagne = probabilities[0][1]
            if proba_equipe1_gagne > proba_equipe2_gagne:
                points[equipe1] += 3
                print(f"{equipe1} gagne, probabilité : {proba_equipe1_gagne}")
            elif proba_equipe2_gagne > proba_equipe1_gagne:
                points[equipe2] += 3
                print(f"{equipe2} gagne, probabilité : {proba_equipe2_gagne}")

            else:
                points[equipe1] += 1
                points[equipe2] += 1
                print(f"Egalité")

        classement_groupe = sorted(points.items(), key=lambda x: x[1], reverse=True)
        equipes_qualifiees_par_groupe[groupe].extend([equipe for equipe, _ in classement_groupe[:2]])

    return equipes_qualifiees_par_groupe


"""

def faire_match(equipe1, equipe2)
    Args : 
        equipe1 : first team
        equipe2 : second team

     return :
         equipe1 : if team 1 won
         equipe2 : if team 2 won
print probabilitie of the winner team
"""
def faire_match(equipe1, equipe2, model, build_match, rankingData, reraBase, scaler):
    print(f"Match: {equipe1} contre {equipe2}")
    match = build_match(rankingData, reraBase, equipe1, equipe2, "FIFA World Cup", True, scaler)
    probabilities = model.predict_proba(match)
    proba_equipe1_gagne = probabilities[0][0]
    proba_equipe2_gagne = probabilities[0][1]

    if proba_equipe1_gagne > proba_equipe2_gagne:
        print(f"{equipe1} gagne, probabilité : {proba_equipe1_gagne}")
        return equipe1, proba_equipe1_gagne

    elif proba_equipe1_gagne < proba_equipe2_gagne:
        print(f"{equipe2} gagne, probabilité : {proba_equipe2_gagne}")
        return equipe2, proba_equipe2_gagne
    else:
        if (random.randint(0, 1)):
            print(f"{equipe1} gagne, probabilité : {proba_equipe1_gagne}")
            return equipe1, proba_equipe1_gagne
        else:
            print(f"{equipe2} gagne, probabilité : {proba_equipe2_gagne}")
            return equipe2, proba_equipe2_gagne


"""
def matches_phase_elimination(equipes_qualifiees_par_groupe)
    Args : 
        equipes_qualifiees_par_groupe : dictionnary returned by the function calculer_points_matchs_par_groupe(groupes) 
    Return :
        Test : list of the teams qualified for the second round of the direct elimination phase
"""
def matches_phase_elimination(equipes_qualifiees_par_groupe, model, build_match, rankingData, reraBase, scaler):
    niveaux = []
    niveaux.append(equipes_qualifiees_par_groupe)
    equipes_gagnantes = {}

    groupes = list(equipes_qualifiees_par_groupe.keys())
    matches = []
    test = []
    inc = 0

    while len(groupes) > 1:
        for i in range(0, len(groupes), 2):
            groupe1 = groupes.pop(0)
            groupe2 = groupes.pop(0)
            match_groupe = (
                equipes_qualifiees_par_groupe[groupe1][0],
                equipes_qualifiees_par_groupe[groupe2][1]
            )
            matches.append(match_groupe)

            match_groupe = (
                equipes_qualifiees_par_groupe[groupe1][1],
                equipes_qualifiees_par_groupe[groupe2][0]
            )
            matches.append(match_groupe)

    for match in matches:
        equipe1, equipe2 = match
        print(f"Match: {equipe1} contre {equipe2}")

        match = build_match(rankingData, reraBase, equipe1, equipe2, "FIFA World Cup", True, scaler)
        probabilities = model.predict_proba(match)
        proba_equipe1_gagne = probabilities[0][0]
        proba_equipe2_gagne = probabilities[0][1]
        if proba_equipe1_gagne > proba_equipe2_gagne:
            test.append(equipe1)
            probabilite_str = "{:.2f}".format(proba_equipe1_gagne)
            interfaceGraphiqueElimDirect.append([equipe1, probabilite_str])
            print(f"{equipe1} gagne, probabilité : {proba_equipe1_gagne}")
        elif proba_equipe1_gagne < proba_equipe2_gagne:
            test.append(equipe2)
            probabilite_str = "{:.2f}".format(proba_equipe2_gagne)
            interfaceGraphiqueElimDirect.append([equipe2, probabilite_str])
            print(f"{equipe2} gagne, probabilité : {proba_equipe2_gagne}")
        else:
            if (random.randint(0, 1)):
                test.append(equipe1)
                probabilite_str = "0.5"
                interfaceGraphiqueElimDirect.append([equipe1, probabilite_str])
                print(f"{equipe1} gagne, probabilité : {proba_equipe1_gagne}")
            else:
                test.append(equipe2)
                probabilite_str = "0.5"
                interfaceGraphiqueElimDirect.append([equipe2, probabilite_str])
                print(f"{equipe2} gagne, probabilité : {proba_equipe2_gagne}")

    return test


"""
Documentation : 
    def suiteTournois(liste) 
        Args : 
            liste : liste returned by the function faire_match(equipe1, equipe2)

        return :
            gagnant : Winner of the championship cup
   Iteration of the function while there is juste 1 team wich is in the list and will be the winner
"""
def suiteTournois(liste, model, build_match, rankingData, reraBase, scaler):
    vainqueurs = []
    while len(liste) > 1:
        prochains_vainqueurs = []
        for i in range(0, len(liste), 2):
            if i + 1 < len(liste):
                gagnant, probaGagnant = faire_match(liste[i], liste[i + 1], model, build_match, rankingData, reraBase, scaler)
                if gagnant:
                    prochains_vainqueurs.append(gagnant)
                    probabilite_str = "{:.2f}".format(probaGagnant)
                    interfaceGraphiqueElimDirect.append([gagnant, probabilite_str])
        liste = prochains_vainqueurs[:]
        vainqueurs.extend(prochains_vainqueurs)
    gagnant = vainqueurs[len(vainqueurs) - 1]
    print(f"le vainqueur du tournois est : {gagnant} !")

def appendGroupPhaseElim (phase_elimination) :
    for groupe, pays in phase_elimination.items():
        for pays_nom in pays:
            interfaceGraphiqueElimDirect.append([pays_nom, groupe])
    return interfaceGraphiqueElimDirect

"""
Documentation : 
    def demarrerTournois(filename)
        Args : 
            filename : name of the csv file to load

        return :
            gagnant : Winner of the championship cup
   Utilisation of all the function to create the Tournament in its globality
"""
def demarrerTournois(filename, model, build_match, rankingData, reraBase, scaler):
    groupe = lire_fichier_csv_groupes(filename)
    print(f"Importation du fichier {filename}")
    print("------------------------------------------------")
    print("The group stage : ")
    print("------------------------------------------------")
    phase_elimination = calculer_points_matchs_par_groupe(groupe, model, build_match, rankingData, reraBase, scaler)
    appendGroupPhaseElim(phase_elimination)
    print("------------------------------------------------")
    print("The knockout stage : ")
    print("------------------------------------------------")
    res = matches_phase_elimination(phase_elimination, model, build_match, rankingData, reraBase, scaler)
    suiteTournois(res, model, build_match, rankingData, reraBase, scaler)

    return len(groupe), interfaceGraphiqueElimDirect

