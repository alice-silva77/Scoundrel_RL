import tkinter as tk
from tkinter import messagebox
import random

# --- CONFIGURATION ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 650
BG_COLOR = "#2e8b57"  # Sea Green
CARD_WIDTH = 100
CARD_HEIGHT = 140

# --- LOGIC CLASSES ---
class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        self.is_red = suit in ['H', 'D']
    
    def get_name(self):
        names = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        return names.get(self.value, str(self.value))

    def get_symbol(self):
        symbols = {'H': '♥', 'D': '♦', 'C': '♣', 'S': '♠'}
        return symbols[self.suit]

    def get_type(self):
        if self.suit == 'H': return "POTION"
        if self.suit == 'D': return "WEAPON"
        return "MONSTER"

class ScoundrelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Scoundrel - Tactical Mode")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=BG_COLOR)

        # Game State
        self.deck = []
        self.room = []
        self.health = 20
        self.max_health = 20
        self.weapon_val = 0
        self.weapon_previous_kill = 999 
        self.cards_played_in_room = 0
        self.potion_drunk_in_room = False
        self.can_run = True 
        self.game_over = False

        self.setup_ui()
        self.start_new_game()

    def setup_ui(self):
        # --- Top Info Bar ---
        self.info_frame = tk.Frame(self.root, bg="#206040", pady=10)
        self.info_frame.pack(fill="x")

        self.lbl_health = tk.Label(self.info_frame, text="HP: 20/20", font=("Arial", 16, "bold"), bg="#206040", fg="white")
        self.lbl_health.pack(side="left", padx=20)

        self.lbl_weapon = tk.Label(self.info_frame, text="Weapon: None", font=("Arial", 14), bg="#206040", fg="#ffd700")
        self.lbl_weapon.pack(side="left", padx=20)

        self.lbl_deck = tk.Label(self.info_frame, text="Deck: 0", font=("Arial", 14), bg="#206040", fg="white")
        self.lbl_deck.pack(side="right", padx=20)

        # --- Message Log ---
        self.lbl_message = tk.Label(self.root, text="Welcome to the Dungeon.", font=("Arial", 14, "italic"), bg=BG_COLOR, fg="white", pady=20)
        self.lbl_message.pack()

        # --- Card Area ---
        self.card_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.card_frame.pack(expand=True)

        self.card_buttons = []
        for i in range(4):
            f = tk.Frame(self.card_frame, width=CARD_WIDTH, height=CARD_HEIGHT, bg=BG_COLOR)
            f.grid(row=0, column=i, padx=15, pady=20)
            f.pack_propagate(False)
            
            btn = tk.Button(f, text="", font=("Arial", 24, "bold"), 
                            command=lambda idx=i: self.on_card_click(idx))
            btn.pack(expand=True, fill="both")
            self.card_buttons.append(btn)

        # --- Footer Controls ---
        self.control_frame = tk.Frame(self.root, bg=BG_COLOR, pady=20)
        self.control_frame.pack(side="bottom", fill="x")

        self.btn_run = tk.Button(self.control_frame, text="RUN", command=self.on_run_click, 
                                 font=("Arial", 12, "bold"), bg="#555", fg="white", width=25)
        self.btn_run.pack(pady=10)

        self.lbl_progress = tk.Label(self.control_frame, text="Cards Played: 0/3", font=("Arial", 12), bg=BG_COLOR, fg="white")
        self.lbl_progress.pack()

    def start_new_game(self):
        self.deck = []
        
        # --- DECK BUILDING (Hard Mode) ---
        # 1. Keep Black Cards (Spades/Clubs) 2-14
        # 2. Keep Red Cards (Hearts/Diamonds) ONLY 2-10.
        # 3. Remove Red Face Cards (11,12,13) AND Red Aces (14).
        
        for suit in ['H', 'D', 'C', 'S']:
            for val in range(2, 15):
                if suit in ['H', 'D']:
                    # Remove Jack(11) through Ace(14)
                    if val >= 11:
                        continue
                self.deck.append(Card(suit, val))
        
        random.shuffle(self.deck)
        
        self.health = 20
        self.weapon_val = 0
        self.weapon_previous_kill = 999
        self.room = []
        self.cards_played_in_room = 0
        self.potion_drunk_in_room = False
        self.can_run = True
        self.game_over = False
        
        self.draw_room()
        self.update_ui()

    def draw_room(self):
        cards_needed = 4 - len(self.room)
        for _ in range(cards_needed):
            if self.deck:
                self.room.append(self.deck.pop(0))

    def update_ui(self):
        self.lbl_health.config(text=f"♥ HP: {self.health}/{self.max_health}")
        
        w_text = "No Weapon"
        if self.weapon_val > 0:
            limit = self.weapon_previous_kill - 1
            limit_text = f"Max Target: {limit}" if self.weapon_previous_kill < 999 else "Target: Any"
            w_text = f"⚔ {self.weapon_val} DMG ({limit_text})"
        self.lbl_weapon.config(text=w_text)
        
        self.lbl_deck.config(text=f"Cards Left: {len(self.deck)}")
        self.lbl_progress.config(text=f"Played in Room: {self.cards_played_in_room}/3")
        
        # Strict Run Logic
        if self.can_run and self.cards_played_in_room == 0:
            self.btn_run.config(state="normal", text="RUN (Shuffle Room)", bg="#d9534f")
        else:
            if not self.can_run:
                msg = "Cannot Run (Ran Previous)"
            else:
                msg = "Room Engaged (Must Clear)"
            self.btn_run.config(state="disabled", text=msg, bg="#555")

        for i, btn in enumerate(self.card_buttons):
            if i < len(self.room):
                card = self.room[i]
                symbol = card.get_symbol()
                name = card.get_name()
                color = "red" if card.is_red else "black"
                display_text = f"{name}\n{symbol}"
                btn.config(text=display_text, fg=color, bg="white", state="normal", relief="raised")
            else:
                btn.config(text="", bg="#257045", state="disabled", relief="flat")

    def on_card_click(self, idx):
        if self.game_over: return
        
        card = self.room[idx]
        ctype = card.get_type()
        msg = ""
        
        if ctype == "POTION":
            if self.potion_drunk_in_room:
                msg = f"Discarded {card.get_name()} potion (Already drank one!)"
            else:
                heal = card.value
                self.health = min(self.max_health, self.health + heal)
                self.potion_drunk_in_room = True
                msg = f"Drank Potion. Healed {heal} HP."

        elif ctype == "WEAPON":
            self.weapon_val = card.value
            self.weapon_previous_kill = 999 
            msg = f"Equipped Weapon: {card.value} Power."

        elif ctype == "MONSTER":
            monster_val = card.value
            damage = 0
            
            # --- UPDATED COMBAT LOGIC ---
            # Check if using the weapon is physically possible by the rules
            can_use_weapon = (self.weapon_val > 0) and (monster_val < self.weapon_previous_kill)
            
            use_weapon = False
            
            if can_use_weapon:
                # Ask the player what they want to do
                use_weapon = messagebox.askyesno(
                    "Combat Choice", 
                    f"Fighting Monster: {monster_val}\n"
                    f"Equipped Weapon: {self.weapon_val}\n\n"
                    f"Do you want to use your WEAPON?\n"
                    f"(YES = Take less damage, Weapon dulls)\n"
                    f"(NO = Take full damage, Save Weapon)"
                )
            
            if use_weapon:
                # Combat with Weapon
                damage = max(0, monster_val - self.weapon_val)
                self.weapon_previous_kill = monster_val # The weapon dulls
                msg = f"Slayed Monster {monster_val} with Weapon. Took {damage} dmg."
            else:
                # Combat Barehanded (Either by choice or necessity)
                damage = monster_val
                msg = f"Fought Monster {monster_val} Barehanded. Took {damage} dmg."
                
            self.health -= damage

        self.room.pop(idx)
        self.cards_played_in_room += 1
        self.lbl_message.config(text=msg)

        if self.health <= 0:
            self.health = 0
            self.update_ui()
            messagebox.showinfo("GAME OVER", "You have died in the dungeon.")
            self.root.quit()
            return

        if self.cards_played_in_room >= 3:
            self.cards_played_in_room = 0
            self.potion_drunk_in_room = False
            self.can_run = True 
            self.draw_room()

        if len(self.deck) == 0 and len(self.room) == 0:
            self.update_ui()
            messagebox.showinfo("VICTORY", "You cleared the dungeon!")
            self.root.quit()
            return

        self.update_ui()

    def on_run_click(self):
        if not self.can_run or self.cards_played_in_room > 0: return
        if len(self.deck) == 0 and len(self.room) < 4:
            self.lbl_message.config(text="Nowhere left to run!")
            return

        self.deck.extend(self.room)
        self.room = []
        self.can_run = False
        self.draw_room()
        self.cards_played_in_room = 0
        self.potion_drunk_in_room = False
        
        self.lbl_message.config(text="You ran away! Next room is inescapable.")
        self.update_ui()

if __name__ == "__main__":
    root = tk.Tk()
    app = ScoundrelGUI(root)
    root.mainloop()