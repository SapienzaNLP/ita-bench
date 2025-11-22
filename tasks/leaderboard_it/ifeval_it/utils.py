##### INSTRUCTION_UTILS.PY

# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility library of instructions"""

import functools
import random
import re

import immutabledict
import nltk


def download_nltk_resources():
    """Download 'punkt' if not already installed"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


download_nltk_resources()

WORD_LIST_EN = [
    "western",
    "sentence",
    "signal",
    "dump",
    "spot",
    "opposite",
    "bottom",
    "potato",
    "administration",
    "working",
    "welcome",
    "morning",
    "good",
    "agency",
    "primary",
    "wish",
    "responsibility",
    "press",
    "problem",
    "president",
    "steal",
    "brush",
    "read",
    "type",
    "beat",
    "trainer",
    "growth",
    "lock",
    "bone",
    "case",
    "equal",
    "comfortable",
    "region",
    "replacement",
    "performance",
    "mate",
    "walk",
    "medicine",
    "film",
    "thing",
    "rock",
    "tap",
    "total",
    "competition",
    "ease",
    "south",
    "establishment",
    "gather",
    "parking",
    "world",
    "plenty",
    "breath",
    "claim",
    "alcohol",
    "trade",
    "dear",
    "highlight",
    "street",
    "matter",
    "decision",
    "mess",
    "agreement",
    "studio",
    "coach",
    "assist",
    "brain",
    "wing",
    "style",
    "private",
    "top",
    "brown",
    "leg",
    "buy",
    "procedure",
    "method",
    "speed",
    "high",
    "company",
    "valuable",
    "pie",
    "analyst",
    "session",
    "pattern",
    "district",
    "pleasure",
    "dinner",
    "swimming",
    "joke",
    "order",
    "plate",
    "department",
    "motor",
    "cell",
    "spend",
    "cabinet",
    "difference",
    "power",
    "examination",
    "engine",
    "horse",
    "dimension",
    "pay",
    "toe",
    "curve",
    "literature",
    "bother",
    "fire",
    "possibility",
    "debate",
    "activity",
    "passage",
    "hello",
    "cycle",
    "background",
    "quiet",
    "author",
    "effect",
    "actor",
    "page",
    "bicycle",
    "error",
    "throat",
    "attack",
    "character",
    "phone",
    "tea",
    "increase",
    "outcome",
    "file",
    "specific",
    "inspector",
    "internal",
    "potential",
    "staff",
    "building",
    "employer",
    "shoe",
    "hand",
    "direction",
    "garden",
    "purchase",
    "interview",
    "study",
    "recognition",
    "member",
    "spiritual",
    "oven",
    "sandwich",
    "weird",
    "passenger",
    "particular",
    "response",
    "reaction",
    "size",
    "variation",
    "a",
    "cancel",
    "candy",
    "exit",
    "guest",
    "condition",
    "fly",
    "price",
    "weakness",
    "convert",
    "hotel",
    "great",
    "mouth",
    "mind",
    "song",
    "sugar",
    "suspect",
    "telephone",
    "ear",
    "roof",
    "paint",
    "refrigerator",
    "organization",
    "jury",
    "reward",
    "engineering",
    "day",
    "possession",
    "crew",
    "bar",
    "road",
    "description",
    "celebration",
    "score",
    "mark",
    "letter",
    "shower",
    "suggestion",
    "sir",
    "luck",
    "national",
    "progress",
    "hall",
    "stroke",
    "theory",
    "offer",
    "story",
    "tax",
    "definition",
    "history",
    "ride",
    "medium",
    "opening",
    "glass",
    "elevator",
    "stomach",
    "question",
    "ability",
    "leading",
    "village",
    "computer",
    "city",
    "grand",
    "confidence",
    "candle",
    "priest",
    "recommendation",
    "point",
    "necessary",
    "body",
    "desk",
    "secret",
    "horror",
    "noise",
    "culture",
    "warning",
    "water",
    "round",
    "diet",
    "flower",
    "bus",
    "tough",
    "permission",
    "week",
    "prompt",
    "connection",
    "abuse",
    "height",
    "save",
    "corner",
    "border",
    "stress",
    "drive",
    "stop",
    "rip",
    "meal",
    "listen",
    "confusion",
    "girlfriend",
    "living",
    "relation",
    "significance",
    "plan",
    "creative",
    "atmosphere",
    "blame",
    "invite",
    "housing",
    "paper",
    "drink",
    "roll",
    "silver",
    "drunk",
    "age",
    "damage",
    "smoke",
    "environment",
    "pack",
    "savings",
    "influence",
    "tourist",

    "rain",
    "post",
    "sign",
    "grandmother",
    "run",
    "profit",
    "push",
    "clerk",
    "final",
    "wine",
    "swim",
    "pause",
    "stuff",
    "singer",
    "funeral",
    "average",
    "source",
    "scene",
    "tradition",
    "personal",
    "snow",
    "nobody",
    "distance",
    "sort",
    "sensitive",
    "animal",
    "major",
    "negotiation",
    "click",
    "mood",
    "period",
    "arrival",
    "expression",
    "holiday",
    "repeat",
    "dust",
    "closet",
    "gold",
    "bad",
    "sail",
    "combination",
    "clothes",
    "emphasis",
    "duty",
    "black",
    "step",
    "school",
    "jump",
    "document",
    "professional",
    "lip",
    "chemical",
    "front",
    "wake",
    "while",
    "inside",
    "watch",
    "row",
    "subject",
    "penalty",
    "balance",
    "possible",
    "adult",
    "aside",
    "sample",
    "appeal",
    "wedding",
    "depth",
    "king",
    "award",
    "wife",
    "blow",
    "site",
    "camp",
    "music",
    "safe",
    "gift",
    "fault",
    "guess",
    "act",
    "shame",
    "drama",
    "capital",
    "exam",
    "stupid",
    "record",
    "sound",
    "swing",
    "novel",
    "minimum",
    "ratio",
    "machine",
    "shape",
    "lead",
    "operation",
    "salary",
    "cloud",
    "affair",
    "hit",
    "chapter",
    "stage",
    "quantity",
    "access",
    "army",
    "chain",
    "traffic",
    "kick",
    "analysis",
    "airport",
    "time",
    "vacation",
    "philosophy",
    "ball",
    "chest",
    "thanks",
    "place",
    "mountain",
    "advertising",
    "red",
    "past",
    "rent",
    "return",
    "tour",
    "house",
    "construction",
    "net",
    "native",
    "war",
    "figure",
    "fee",
    "spray",
    "user",
    "dirt",
    "shot",
    "task",
    "stick",
    "friend",
    "software",
    "promotion",
    "interaction",
    "surround",
    "block",
    "purpose",
    "practice",
    "conflict",
    "routine",
    "requirement",
    "bonus",
    "hole",
    "state",
    "junior",
    "sweet",
    "catch",
    "tear",
    "fold",
    "wall",
    "editor",
    "life",
    "position",
    "pound",
    "respect",
    "bathroom",
    "coat",
    "script",
    "job",
    "teach",
    "birth",
    "view",
    "resolve",
    "theme",
    "employee",
    "doubt",
    "market",
    "education",
    "serve",
    "recover",
    "tone",
    "harm",
    "miss",
    "union",
    "understanding",
    "cow",
    "river",
    "association",
    "concept",
    "training",
    "recipe",
    "relationship",
    "reserve",
    "depression",
    "proof",
    "hair",
    "revenue",
    "independent",
    "lift",
    "assignment",
    "temporary",
    "amount",
    "loss",
    "edge",
    "track",
    "check",
    "rope",
    "estimate",
    "pollution",
    "stable",
    "message",
    "delivery",
    "perspective",
    "mirror",
    "assistant",
    "representative",
    "witness",
    "nature",
    "judge",
    "fruit",
    "tip",
    "devil",
    "town",
    "emergency",
    "upper",
    "drop",
    "stay",
    "human",
    "neck",
    "speaker",
    "network",
    "sing",
    "resist",
    "league",
    "trip",
    "signature",
    "lawyer",
    "importance",
    "gas",
    "choice",
    "engineer",
    "success",
    "part",
    "external",
    "worker",
    "simple",
    "quarter",
    "student",
    "heart",
    "pass",
    "spite",
    "shift",
    "rough",
    "lady",
    "grass",
    "community",
    "garage",
    "youth",
    "standard",
    "skirt",
    "promise",
    "blind",
    "television",
    "disease",
    "commission",
    "positive",
    "energy",
    "calm",
    "presence",
    "tune",
    "basis",
    "preference",
    "head",
    "common",
    "cut",
    "somewhere",
    "presentation",
    "current",
    "thought",
    "revolution",
    "effort",
    "master",
    "implement",
    "republic",
    "floor",
    "principle",
    "stranger",
    "shoulder",
    "grade",
    "button",
    "tennis",
    "police",
    "collection",
    "account",
    "register",
    "glove",
    "divide",
    "professor",
    "chair",
    "priority",
    "combine",
    "peace",
    "extension",
    "maybe",

    "evening",
    "frame",
    "sister",
    "wave",
    "code",
    "application",
    "mouse",
    "match",
    "counter",
    "bottle",
    "half",
    "cheek",
    "resolution",
    "back",
    "knowledge",
    "make",
    "discussion",
    "screw",
    "length",
    "accident",
    "battle",
    "dress",
    "knee",
    "log",
    "package",
    "it",
    "turn",
    "hearing",
    "newspaper",
    "layer",
    "wealth",
    "profile",
    "imagination",
    "answer",
    "weekend",
    "teacher",
    "appearance",
    "meet",
    "bike",
    "rise",
    "belt",
    "crash",
    "bowl",
    "equivalent",
    "support",
    "image",
    "poem",
    "risk",
    "excitement",
    "remote",
    "secretary",
    "public",
    "produce",
    "plane",
    "display",
    "money",
    "sand",
    "situation",
    "punch",
    "customer",
    "title",
    "shake",
    "mortgage",
    "option",
    "number",
    "pop",
    "window",
    "extent",
    "nothing",
    "experience",
    "opinion",
    "departure",
    "dance",
    "indication",
    "boy",
    "material",
    "band",
    "leader",
    "sun",
    "beautiful",
    "muscle",
    "farmer",
    "variety",
    "fat",
    "handle",
    "director",
    "opportunity",
    "calendar",
    "outside",
    "pace",
    "bath",
    "fish",
    "consequence",
    "put",
    "owner",
    "go",
    "doctor",
    "information",
    "share",
    "hurt",
    "protection",
    "career",
    "finance",
    "force",
    "golf",
    "garbage",
    "aspect",
    "kid",
    "food",
    "boot",
    "milk",
    "respond",
    "objective",
    "reality",
    "raw",
    "ring",
    "mall",
    "one",
    "impact",
    "area",
    "news",
    "international",
    "series",
    "impress",
    "mother",
    "shelter",
    "strike",
    "loan",
    "month",
    "seat",
    "anything",
    "entertainment",
    "familiar",
    "clue",
    "year",
    "glad",
    "supermarket",
    "natural",
    "god",
    "cost",
    "conversation",
    "tie",
    "ruin",
    "comfort",
    "earth",
    "storm",
    "percentage",
    "assistance",
    "budget",
    "strength",
    "beginning",
    "sleep",
    "other",
    "young",
    "unit",
    "fill",
    "store",
    "desire",
    "hide",
    "value",
    "cup",
    "maintenance",
    "nurse",
    "function",
    "tower",
    "role",
    "class",
    "camera",
    "database",
    "panic",
    "nation",
    "basket",
    "ice",
    "art",
    "spirit",
    "chart",
    "exchange",
    "feedback",
    "statement",
    "reputation",
    "search",
    "hunt",
    "exercise",
    "nasty",
    "notice",
    "male",
    "yard",
    "annual",
    "collar",
    "date",
    "platform",
    "plant",
    "fortune",
    "passion",
    "friendship",
    "spread",
    "cancer",
    "ticket",
    "attitude",
    "island",
    "active",
    "object",
    "service",
    "buyer",
    "bite",
    "card",
    "face",
    "steak",
    "proposal",
    "patient",
    "heat",
    "rule",
    "resident",
    "broad",
    "politics",
    "west",
    "knife",
    "expert",
    "girl",
    "design",
    "salt",
    "baseball",
    "grab",
    "inspection",
    "cousin",
    "couple",
    "magazine",
    "cook",
    "dependent",
    "security",
    "chicken",
    "version",
    "currency",
    "ladder",
    "scheme",
    "kitchen",
    "employment",
    "local",
    "attention",
    "manager",
    "fact",
    "cover",
    "sad",
    "guard",
    "relative",
    "county",
    "rate",
    "lunch",
    "program",
    "initiative",
    "gear",
    "bridge",
    "breast",
    "talk",
    "dish",
    "guarantee",
    "beer",
    "vehicle",
    "reception",
    "woman",
    "substance",
    "copy",
    "lecture",
    "advantage",
    "park",
    "cold",
    "death",
    "mix",
    "hold",
    "scale",
    "tomorrow",
    "blood",
    "request",
    "green",
    "cookie",
    "church",
    "strip",
    "forever",
    "beyond",
    "debt",
    "tackle",
    "wash",
    "following",
    "feel",
    "maximum",
    "sector",
    "sea",
    "property",
    "economics",
    "menu",
    "bench",
    "try",
    "language",
    "start",
    "call",
    "solid",
    "address",
    "income",
    "foot",
    "senior",
    "honey",
    "few",
    "mixture",
    "cash",
    "grocery",
    "link",
    "map",
    "form",
    "factor",
    "pot",
    "model",
    "writer",
    "farm",
    "winter",
    "skill",
    "anywhere",
    "birthday",
    "policy",
    "release",
    "husband",
    "lab",
    "hurry",
    "mail",
    "equipment",
    "sink",
    "pair",
    "driver",
    "consideration",
    "leather",
    "skin",
    "blue",
    "boat",
    "sale",
    "brick",
    "two",
    "feed",
    "square",
    "dot",
    "rush",
    "dream",
    "location",
    "afternoon",
    "manufacturer",
    "control",
    "occasion",
    "trouble",
    "introduction",
    "advice",
    "bet",
    "eat",
    "kill",
    "category",
    "manner",
    "office",
    "estate",
    "pride",
    "awareness",
    "slip",
    "crack",
    "client",
    "nail",
    "shoot",
    "membership",
    "soft",
    "anybody",
    "web",
    "official",
    "individual",
    "pizza",
    "interest",
    "bag",
    "spell",
    "profession",
    "queen",
    "deal",
    "resource",
    "ship",
    "guy",
    "chocolate",
    "joint",
    "formal",
    "upstairs",
    "car",
    "resort",
    "abroad",
    "dealer",
    "associate",
    "finger",
    "surgery",
    "comment",
    "team",
    "detail",
    "crazy",
    "path",
    "tale",
    "initial",
    "arm",
    "radio",
    "demand",
    "single",

    "draw",
    "yellow",
    "contest",
    "piece",
    "quote",
    "pull",
    "commercial",
    "shirt",
    "contribution",
    "cream",
    "channel",
    "suit",
    "discipline",
    "instruction",
    "concert",
    "speech",
    "low",
    "effective",
    "hang",
    "scratch",
    "industry",
    "breakfast",
    "lay",
    "join",
    "metal",
    "bedroom",
    "minute",
    "product",
    "rest",
    "temperature",
    "many",
    "give",
    "argument",
    "print",
    "purple",
    "laugh",
    "health",
    "credit",
    "investment",
    "sell",
    "setting",
    "lesson",
    "egg",
    "middle",
    "marriage",
    "level",
    "evidence",
    "phrase",
    "love",
    "self",
    "benefit",
    "guidance",
    "affect",
    "you",
    "dad",
    "anxiety",
    "special",
    "boyfriend",
    "test",
    "blank",
    "payment",
    "soup",
    "obligation",
    "reply",
    "smile",
    "deep",
    "complaint",
    "addition",
    "review",
    "box",
    "towel",
    "minor",
    "fun",
    "soil",
    "issue",
    "cigarette",
    "internet",
    "gain",
    "tell",
    "entry",
    "spare",
    "incident",
    "family",
    "refuse",
    "branch",
    "can",
    "pen",
    "grandfather",
    "constant",
    "tank",
    "uncle",
    "climate",
    "ground",
    "volume",
    "communication",
    "kind",
    "poet",
    "child",
    "screen",
    "mine",
    "quit",
    "gene",
    "lack",
    "charity",
    "memory",
    "tooth",
    "fear",
    "mention",
    "marketing",
    "reveal",
    "reason",
    "court",
    "season",
    "freedom",
    "land",
    "sport",
    "audience",
    "classroom",
    "law",
    "hook",
    "win",
    "carry",
    "eye",
    "smell",
    "distribution",
    "research",
    "country",
    "dare",
    "hope",
    "whereas",
    "stretch",
    "library",
    "if",
    "delay",
    "college",
    "plastic",
    "book",
    "present",
    "use",
    "worry",
    "champion",
    "goal",
    "economy",
    "march",
    "election",
    "reflection",
    "midnight",
    "slide",
    "inflation",
    "action",
    "challenge",
    "guitar",
    "coast",
    "apple",
    "campaign",
    "field",
    "jacket",
    "sense",
    "way",
    "visual",
    "remove",
    "weather",
    "trash",
    "cable",
    "regret",
    "buddy",
    "beach",
    "historian",
    "courage",
    "sympathy",
    "truck",
    "tension",
    "permit",
    "nose",
    "bed",
    "son",
    "person",
    "base",
    "meat",
    "usual",
    "air",
    "meeting",
    "worth",
    "game",
    "independence",
    "physical",
    "brief",
    "play",
    "raise",
    "board",
    "she",
    "key",
    "writing",
    "pick",
    "command",
    "party",
    "yesterday",
    "spring",
    "candidate",
    "physics",
    "university",
    "concern",
    "development",
    "change",
    "string",
    "target",
    "instance",
    "room",
    "bitter",
    "bird",
    "football",
    "normal",
    "split",
    "impression",
    "wood",
    "long",
    "meaning",
    "stock",
    "cap",
    "leadership",
    "media",
    "ambition",
    "fishing",
    "essay",
    "salad",
    "repair",
    "today",
    "designer",
    "night",
    "bank",
    "drawing",
    "inevitable",
    "phase",
    "vast",
    "chip",
    "anger",
    "switch",
    "cry",
    "twist",
    "personality",
    "attempt",
    "storage",
    "being",
    "preparation",
    "bat",
    "selection",
    "white",
    "technology",
    "contract",
    "side",
    "section",
    "station",
    "till",
    "structure",
    "tongue",
    "taste",
    "truth",
    "difficulty",
    "group",
    "limit",
    "main",
    "move",
    "feeling",
    "light",
    "example",
    "mission",
    "might",
    "wait",
    "wheel",
    "shop",
    "host",
    "classic",
    "alternative",
    "cause",
    "agent",
    "consist",
    "table",
    "airline",
    "text",
    "pool",
    "craft",
    "range",
    "fuel",
    "tool",
    "partner",
    "load",
    "entrance",
    "deposit",
    "hate",
    "article",
    "video",
    "summer",
    "feature",
    "extreme",
    "mobile",
    "hospital",
    "flight",
    "fall",
    "pension",
    "piano",
    "fail",
    "result",
    "rub",
    "gap",
    "system",
    "report",
    "suck",
    "ordinary",
    "wind",
    "nerve",
    "ask",
    "shine",
    "note",
    "line",
    "mom",
    "perception",
    "brother",
    "reference",
    "bend",
    "charge",
    "treat",
    "trick",
    "term",
    "homework",
    "bake",
    "bid",
    "status",
    "project",
    "strategy",
    "orange",
    "let",
    "enthusiasm",
    "parent",
    "concentrate",
    "device",
    "travel",
    "poetry",
    "business",
    "society",
    "kiss",
    "end",
    "vegetable",
    "employ",
    "schedule",
    "hour",
    "brave",
    "focus",
    "process",
    "movie",
    "illegal",
    "general",
    "coffee",
    "ad",
    "highway",
    "chemistry",
    "psychology",
    "hire",
    "bell",
    "conference",
    "relief",
    "show",
    "neat",
    "funny",
    "weight",
    "quality",
    "club",
    "daughter",
    "zone",
    "touch",
    "tonight",
    "shock",
    "burn",
    "excuse",
    "name",
    "survey",
    "landscape",
    "advance",
    "satisfaction",
    "bread",
    "disaster",
    "item",
    "hat",
    "prior",
    "shopping",
    "visit",
    "east",
    "photo",
    "home",
    "idea",
    "father",
    "comparison",
    "cat",
    "pipe",
    "winner",
    "count",
    "lake",
    "fight",
    "prize",
    "foundation",
    "dog",
    "keep",
    "ideal",
    "fan",
    "struggle",
    "peak",
    "safety",
    "solution",
    "hell",
    "conclusion",
    "population",
    "strain",
    "alarm",
    "measurement",
    "second",
    "train",
    "race",
    "due",
    "insurance",
    "boss",
    "tree",
    "monitor",
    "sick",
    "course",
    "drag",
    "appointment",
    "slice",
    "still",
    "care",
    "patience",
    "rich",
    "escape",
    "emotion",
    "royal",
    "female",
    "childhood",
    "government",
    "picture",
    "will",
    "sock",
    "big",
    "gate",
    "oil",
    "cross",
    "pin",
    "improvement",
    "championship",
    "silly",
    "help",
    "sky",
    "pitch",
    "man",
    "diamond",
    "most",
    "transition",
    "work",
    "science",
    "committee",
    "moment",
    "fix",
    "teaching",
    "dig",
    "specialist",
    "complex",
    "guide",
    "people",
    "dead",
    "voice",
    "original",
    "break",
    "topic",
    "data",
    "degree",
    "reading",
    "recording",
    "bunch",
    "reach",
    "judgment",
    "lie",
    "regular",
    "set",
    "painting",
    "mode",
    "list",
    "player",
    "bear",
    "north",
    "wonder",
    "carpet",
    "heavy",
    "officer",
    "negative",
    "clock",
    "unique",
    "baby",
    "pain",
    "assumption",
    "disk",
    "iron",
    "bill",
    "drawer",

    "look",
    "double",
    "mistake",
    "finish",
    "future",
    "brilliant",
    "contact",
    "math",
    "rice",
    "leave",
    "restaurant",
    "discount",
    "sex",
    "virus",
    "bit",
    "trust",
    "event",
    "wear",
    "juice",
    "failure",
    "bug",
    "context",
    "mud",
    "whole",
    "wrap",
    "intention",
    "draft",
    "pressure",
    "cake",
    "dark",
    "explanation",
    "space",
    "angle",
    "word",
    "efficiency",
    "management",
    "habit",
    "star",
    "chance",
    "finding",
    "transportation",
    "stand",
    "criticism",
    "flow",
    "door",
    "injury",
    "insect",
    "surprise",
    "apartment",
]  # pylint: disable=line-too-long

WORD_LIST_IT = [
    "occidentale",
    "frase", "sentenza",
    "segnale",
    "discarica", "scaricare",
    "posto", "macchia", "punto",
    "opposto", "di fronte",
    "fondo", "parte inferiore",
    "patata",
    "amministrazione", "governo", "gestione",
    "lavoro", "funzionante",
    "benvenuto",
    "mattina",
    "buono", "bene",
    "agenzia", "ente",
    "primario", "principale",
    "desiderio", "augurio",
    "responsabilità", "compito",
    "stampa", "premere",
    "problema",
    "presidente",
    "rubare",
    "spazzola", "spazzolare",
    "leggere",
    "tipo", "digitare",
    "battere", "sconfiggere",
    "allenatore", "istruttore",
    "crescita", "sviluppo",
    "lucchetto", "chiudere a chiave",
    "osso",
    "caso", "scatola",
    "uguale", "pari",
    "comodo", "confortevole",
    "regione", "zona",
    "sostituzione", "rimpiazzo",
    "prestazione", "esecuzione",
    "compagno", "partner",
    "camminare", "passeggiata",
    "medicina", "farmaco",
    "film", "pellicola",
    "cosa", "oggetto",
    "roccia", "sasso", "rock",
    "toccare", "rubinetto",
    "totale", "complessivo",
    "competizione", "gara",
    "facilità", "agio",
    "sud",
    "fondazione", "stabilimento", "locale",
    "raccogliere", "radunare",
    "parcheggio",
    "mondo",
    "abbondanza", "molto",
    "respiro",
    "rivendicare", "affermazione",
    "alcol",
    "commercio", "scambio",
    "caro", "gentile",
    "evidenziare", "punto saliente",
    "strada", "via",
    "questione", "materia", "sostanza",
    "decisione",
    "disordine", "casino",
    "accordo", "patto", "contratto",
    "studio", "atelier", "sala di registrazione",
    "allenatore", "coach",
    "assistere", "aiutare",
    "cervello",
    "ala",
    "stile",
    "privato",
    "cima", "superiore",
    "marrone",
    "gamba", "zampa",
    "comprare", "acquistare",
    "procedura",
    "metodo",
    "velocità",

    "alto", "sballo",
    "azienda", "compagnia", "società",
    "prezioso", "di valore",
    "torta", "crostata",
    "analista",
    "sessione", "seduta",
    "schema", "modello", "motivo",
    "distretto", "quartiere",
    "piacere", "godimento",
    "cena",
    "nuoto",
    "scherzo", "barzelletta",
    "ordine", "ordinare",
    "piatto", "lastra",
    "dipartimento", "reparto",
    "motore",
    "cella", "cellulare",
    "spendere", "trascorrere",
    "armadio", "gabinetto",
    "differenza",
    "potere", "energia", "forza",
    "esame", "esaminazione",
    "motore",
    "cavallo",
    "dimensione", "misura",
    "pagare", "paga",
    "dito del piede", "alluce",
    "curva",
    "letteratura",
    "disturbare", "seccare",
    "fuoco", "incendio", "sparare",
    "possibilità",
    "dibattito", "discussione",
    "attività",
    "passaggio", "corridoio", "brano",
    "ciao", "salve",
    "ciclo", "giro",
    "sfondo", "retroscena",
    "silenzioso", "tranquillo",
    "autore", "scrittore",
    "effetto",
    "attore",
    "pagina",
    "bicicletta",
    "errore",
    "gola",
    "attacco", "aggredire",
    "personaggio", "carattere",
    "telefono",
    "tè",
    "aumento", "crescita",
    "risultato", "esito",
    "file", "schedario", "fascicolo",
    "specifico", "particolare",
    "ispettore",
    "interno",
    "potenziale",
    "personale", "staff",
    "edificio", "costruzione",
    "datore di lavoro",
    "scarpa",
    "mano",
    "direzione", "orientamento",
    "giardino",
    "acquisto", "comprare",
    "colloquio", "intervista",
    "studio", "ricerca",
    "riconoscimento",
    "membro", "iscritto",
    "spirituale",
    "forno",
    "panino", "sandwich",
    "strano", "bizzarro",
    "passeggero",
    "particolare", "specifico",
    "risposta",
    "reazione",
    "dimensione", "taglia",
    "variazione",
    "un", "una",
    "annullare", "cancellare",
    "caramella",
    "uscita",
    "ospite",
    "condizione", "stato",
    "volare", "mosca",
    "prezzo",
    "debolezza",
    "convertire", "trasformare",
    "hotel", "albergo",
    "grande", "ottimo",
    "bocca",
    "mente",
    "canzone", "brano",
    "zucchero",
    "sospettare", "sospetto",
    "telefono", "apparecchio telefonico",
    "orecchio",
    "tetto",
    "vernice", "dipingere",
    "frigorifero",
    "organizzazione", "ente",
    "giuria",
    "ricompensa", "premio",
    "ingegneria",
    "giorno",
    "possesso", "bene",
    "equipaggio",
    "bar", "locale",
    "strada",
    "descrizione",
    "celebrazione", "festeggiamento",
    "punteggio", "partitura",
    "segno", "voto", "marchio",
    "lettera",
    "doccia",
    "suggerimento", "proposta",
    "signore", "sir",
    "fortuna",
    "nazionale",
    "progresso", "avanzamento",
    "sala", "atrio",
    "colpo", "ictus",
    "teoria",
    "offerta", "proposta",
    "storia", "racconto",
    "tassa",
    "definizione",
    "storia", "storia passata",
    "cavalcare", "corsa", "giro",
    "mezzo", "medio",
    "apertura", "inizio",
    "vetro", "bicchiere",
    "ascensore",
    "stomaco", "pancia",
    "domanda", "quesito",
    "abilità", "capacità",
    "leader", "principale",
    "villaggio", "paese",
    "computer",
    "città",
    "grande", "maestoso",
    "fiducia", "sicurezza di sé",
    "candela",
    "prete", "sacerdote",
    "raccomandazione", "consiglio",
    "punto", "segnalare",
    "necessario", "indispensabile",
    "corpo",
    "scrivania", "banco",
    "segreto",
    "orrore", "paura",
    "rumore",
    "cultura",
    "avvertimento", "avviso",
    "acqua",
    "rotondo", "giro",
    "dieta",
    "fiore",
    "autobus", "bus",
    "duro", "tosto", "difficile",
    "permesso", "autorizzazione",
    "settimana",
    "pronto", "sollecito",
    "connessione", "collegamento",
    "abuso", "maltrattamento",
    "altezza",
    "salvare", "risparmiare",
    "angolo", "spigolo",
    "confine", "frontiera",
    "stress", "tensione",
    "guidare", "corsa",
    "fermarsi", "stop",
    "strappare", "lacerare",
    "pasto",
    "ascoltare",
    "confusione",
    "fidanzata",
    "vivere", "abitazione",
    "relazione", "parentela",
    "importanza", "significato",
    "piano", "programma",
    "creativo",
    "atmosfera", "ambiente",
    "colpa",
    "invitare", "invito",
    "alloggio", "abitazione",
    "carta", "giornale",
    "bevanda", "bere",
    "rotolo", "rotolare",
    "argento",
    "ubriaco",
    "età",
    "danno",
    "fumo", "fumare",
    "ambiente",
    "pacchetto", "confezione",
    "risparmi",
    "influenza", "influsso",
    "turista",

    "pioggia",
    "posta", "palo", "inviare",
    "segno", "cartello", "firmare",
    "nonna",
    "correre", "gestire",
    "profitto", "guadagno",
    "spingere", "pressione",
    "impiegato", "commesso",
    "finale", "definitivo",
    "vino",
    "nuotare",
    "pausa", "interruzione",
    "roba", "materiale",
    "cantante",
    "funerale",
    "media", "medio",
    "fonte", "origine",
    "scena", "scenario",
    "tradizione",
    "personale", "privato",
    "neve",
    "nessuno",
    "distanza",
    "tipo", "ordinare",
    "sensibile", "delicato",
    "animale",
    "maggiore", "importante",
    "negoziazione", "trattativa",
    "clic", "scatto",
    "umore",
    "periodo", "epoca",
    "arrivo",
    "espressione",
    "vacanza", "festività",
    "ripetere",
    "polvere",
    "armadio", "ripostiglio",
    "oro",
    "cattivo",
    "vela", "navigare",
    "combinazione", "unione",
    "vestiti", "abiti",
    "enfasi", "accento",
    "dovere", "obbligo",
    "nero",
    "passo", "scalino",
    "scuola",
    "saltare",
    "documento",
    "professionale",
    "labbro",
    "chimico",
    "fronte", "davanti",
    "sveglia", "svegliarsi",
    "mentre",
    "dentro",
    "orologio", "guardare",
    "fila", "riga", "remare",
    "soggetto", "materia",
    "penalità", "sanzione",
    "equilibrio", "bilancia",
    "possibile",
    "adulto",
    "da parte", "a lato",
    "campione", "esempio",
    "appello", "ricorso",
    "matrimonio", "nozze",
    "profondità",
    "re",
    "premio", "riconoscimento",
    "moglie",
    "soffiare", "colpo",
    "sito", "luogo",
    "campo", "accampamento",
    "musica",
    "sicuro", "cassaforte",
    "regalo", "dono",
    "colpa", "guasto",
    "indovinare", "supporre",
    "atto", "agire",
    "vergogna",
    "dramma",
    "capitale",
    "esame", "prova",
    "stupido",
    "record", "registrazione", "primato",
    "suono",
    "altalena", "oscillare",
    "romanzo",
    "minimo",
    "rapporto", "proporzione",
    "macchina",
    "forma", "sagoma",
    "piombo", "guidare", "condurre",
    "operazione", "intervento",
    "stipendio", "salario",
    "nuvola",
    "affare", "relazione",
    "colpire", "successo",
    "capitolo",
    "palcoscenico", "fase",
    "quantità",
    "accesso",
    "esercito",
    "catena",
    "traffico",
    "calcio", "colpo",
    "analisi",
    "aeroporto",
    "tempo", "ora",
    "vacanza",
    "filosofia",
    "palla", "ballo",
    "petto", "cassa",
    "grazie",
    "posto", "luogo",
    "montagna",
    "pubblicità",
    "rosso",
    "passato",
    "affitto", "noleggio",
    "ritorno", "rendere",
    "tour", "giro",
    "casa",
    "costruzione", "cantiere",
    "rete", "netto",
    "nativo", "indigeno",
    "guerra",
    "figura", "numero",
    "tassa", "compenso",
    "spruzzare", "spruzzo",
    "utente",
    "sporcizia", "terra",
    "colpo", "sparare",
    "compito",
    "bastone", "attaccare",
    "amico",
    "software",
    "promozione", "avanzamento",
    "interazione",
    "circondare",
    "blocco", "isolato",
    "scopo", "finalità",
    "pratica", "allenamento",
    "conflitto",
    "routine", "abitudine",
    "requisito",
    "bonus", "premio",
    "buco",
    "stato", "condizione",
    "junior", "giovane",
    "dolce", "caramella",
    "afferrare", "catturare",
    "lacrima", "strappare",
    "piegare", "piega",
    "muro", "parete",
    "redattore", "editor",
    "vita",
    "posizione", "carica",
    "libbra", "martellare",
    "rispetto",
    "bagno",
    "cappotto", "mantello",
    "copione", "script",
    "lavoro", "impiego",
    "insegnare",
    "nascita",
    "vista", "opinione",
    "risolvere",
    "tema", "argomento",
    "dipendente",
    "dubbio",
    "mercato",
    "istruzione", "educazione",
    "servire",
    "recuperare",
    "tono",
    "danno", "male",
    "perdere", "sentire la mancanza",
    "unione", "sindacato",
    "comprensione", "intesa",
    "mucca",
    "fiume",
    "associazione",
    "concetto",
    "formazione", "addestramento",
    "ricetta",
    "relazione",
    "riserva", "prenotare",
    "depressione", "avvallamento",
    "prova", "dimostrazione",
    "capelli", "pelo",
    "entrate", "ricavi",
    "indipendente",
    "sollevare", "ascensore",
    "compito", "assegnazione",
    "temporaneo", "provvisorio",
    "ammontare", "quantità",
    "perdita",
    "bordo", "orlo",
    "traccia", "pista",
    "controllare", "assegno",
    "corda", "fune",
    "stima", "preventivo",
    "inquinamento",
    "stabile", "scuderia",
    "messaggio",
    "consegna", "spedizione",
    "prospettiva", "punto di vista",
    "specchio",
    "assistente",
    "rappresentante",
    "testimone",
    "natura",
    "giudice",
    "frutto",
    "punta", "mancia", "consiglio",
    "diavolo",
    "cittadina", "paese",
    "emergenza",
    "superiore", "alto",
    "goccia", "lasciare",
    "soggiorno", "restare",
    "umano",
    "collo",
    "altoparlante", "oratore",
    "rete",
    "cantare",
    "resistere",
    "lega", "campionato",
    "viaggio", "gita",
    "firma",
    "avvocato",
    "importanza",
    "gas",
    "scelta",
    "ingegnere",
    "successo",
    "parte",
    "esterno",
    "lavoratore",
    "semplice",
    "quarto", "trimestre",
    "studente",
    "cuore",
    "passare", "biglietto d’ingresso",
    "dispetto",
    "turno", "spostamento",
    "ruvido", "difficile",
    "signora", "dama",
    "erba", "prato",
    "comunità",
    "garage",
    "gioventù", "giovane età",
    "standard",
    "gonna",
    "promessa",
    "cieco",
    "televisione",
    "malattia",
    "commissione", "incarico",
    "positivo",
    "energia",
    "calmo", "tranquillo",
    "presenza",
    "melodia", "accordare",
    "base", "fondamento",
    "preferenza",
    "testa", "capo",
    "comune", "ordinario",
    "tagliare", "ferita",
    "da qualche parte",
    "presentazione", "relazione",
    "corrente", "attuale",
    "pensiero",
    "rivoluzione",
    "sforzo",
    "maestro", "padrone",
    "attuare", "strumento",
    "repubblica",
    "pavimento", "piano",
    "principio",
    "sconosciuto", "estraneo",
    "spalla",
    "voto", "grado",
    "bottone",
    "tennis",
    "polizia",
    "collezione", "raccolta",
    "conto", "account",
    "registro", "registrare",
    "guanto",
    "dividere",
    "professore",
    "sedia", "poltrona",
    "priorità",
    "combinare", "unire",
    "pace",
    "estensione", "prolungamento",
    "forse",

    "sera",
    "cornice", "struttura", "fotogramma",
    "sorella",
    "onda", "salutare",
    "codice",
    "applicazione", "domanda",
    "topo", "mouse",
    "partita", "incontro", "fiammifero", "abbinare",
    "bancone", "contatore",
    "bottiglia",
    "metà",
    "guancia",
    "risoluzione", "determinazione",
    "schiena", "indietro", "retro",
    "conoscenza", "sapere",
    "fare",
    "discussione",
    "vite", "avvitare",
    "lunghezza",
    "incidente",
    "battaglia",
    "vestito", "abito",
    "ginocchio",
    "registro", "tronco",
    "pacchetto", "confezione",
    "esso", "ciò",
    "girare", "svolta",
    "udienza", "udito",
    "giornale",
    "strato",
    "ricchezza",
    "profilo",
    "immaginazione",
    "risposta",
    "fine settimana", "weekend",
    "insegnante", "maestro",
    "aspetto", "apparizione",
    "incontrare",
    "bicicletta", "moto",
    "alzarsi", "sorgere",
    "cintura",
    "incidente", "schianto",
    "ciotola",
    "equivalente",
    "supporto", "sostegno",
    "immagine",
    "poesia",
    "rischio",
    "eccitazione", "entusiasmo",
    "remoto", "telecomando",
    "segretaria", "segretario",
    "pubblico",
    "produrre", "prodotti",
    "aereo", "piano",
    "schermo", "mostrare",
    "denaro", "soldi",
    "sabbia",
    "situazione",
    "pugno", "pugnalata",
    "cliente",
    "titolo",
    "scuotere", "stretta di mano",
    "mutuo", "ipoteca",
    "opzione",
    "numero",
    "scoppio", "pop",
    "finestra",
    "estensione", "portata",
    "niente",
    "esperienza",
    "opinione",
    "partenza",
    "danza", "ballo",
    "indicazione", "segno",
    "ragazzo",
    "materiale",
    "banda", "fascia", "gruppo musicale",
    "leader", "capo",
    "sole",
    "bello", "bella",
    "muscolo",
    "contadino", "agricoltore",
    "varietà",
    "grasso",
    "manico", "gestire",
    "direttore", "regista",
    "opportunità",
    "calendario",
    "fuori", "esterno",
    "andatura", "ritmo",
    "bagno",
    "pesce", "pescare",
    "conseguenza",
    "mettere",
    "proprietario",
    "andare",
    "medico", "dottore",
    "informazioni",
    "condividere", "quota",
    "ferire", "male",
    "protezione",
    "carriera",
    "finanza",
    "forza",
    "golf",
    "spazzatura", "immondizia",
    "aspetto", "punto di vista",
    "bambino", "ragazzino",
    "cibo",
    "stivale",
    "latte",
    "rispondere",
    "obiettivo", "scopo",
    "realtà",
    "crudo", "grezzo",
    "anello", "squillo",
    "centro commerciale",
    "uno",
    "impatto", "influenza",
    "area", "zona",
    "notizie", "news",
    "internazionale",
    "serie",
    "impressionare",
    "madre",
    "rifugio", "shelter",
    "sciopero", "colpire",
    "prestito",
    "mese",
    "sedile", "posto",
    "qualsiasi cosa",
    "intrattenimento",
    "familiare", "conosciuto",
    "indizio",
    "anno",
    "contento", "lieto",
    "supermercato",
    "naturale",
    "dio",
    "costo",
    "conversazione",
    "cravatta", "legare",
    "rovina", "distruggere",
    "comodità", "conforto",
    "terra",
    "tempesta",
    "percentuale",
    "assistenza", "aiuto",
    "bilancio",
    "forza", "vigore",
    "inizio",
    "sonno", "dormire",
    "altro",
    "giovane",
    "unità",
    "riempire",
    "negozio", "conservare",
    "desiderio",
    "nascondere",
    "valore",
    "tazza", "coppa",
    "manutenzione",
    "infermiera",
    "funzione",
    "torre",
    "ruolo",
    "classe", "lezione",
    "fotocamera", "telecamera",
    "database",
    "panico",
    "nazione",
    "cesto", "canestro",
    "ghiaccio",
    "arte",
    "spirito", "anima",
    "grafico",
    "scambio",
    "feedback", "riscontro",
    "dichiarazione",
    "reputazione",
    "ricerca",
    "caccia",
    "esercizio",
    "cattivo", "sgradevole",
    "avviso", "notare",
    "maschio",
    "cortile", "giardino",
    "annuale",
    "collare", "colletto",
    "data", "appuntamento", "dattero",
    "piattaforma", "banchina",
    "pianta", "fabbrica",
    "fortuna", "ricchezza",
    "passione",
    "amicizia",
    "diffondere", "spalmare",
    "cancro",
    "biglietto", "ticket",
    "atteggiamento",
    "isola",
    "attivo",
    "oggetto", "opporre",
    "servizio",
    "acquirente", "compratore",
    "morso", "mordere",
    "carta", "tessera",
    "faccia", "viso",
    "bistecca",
    "proposta",
    "paziente", "malato",
    "calore", "riscaldare",
    "regola", "dominare",
    "residente", "abitante",
    "ampio", "largo",
    "politica",
    "ovest", "occidente",
    "coltello",
    "esperto",
    "ragazza",
    "disegno", "progettare",
    "sale",
    "baseball",
    "afferrare", "prendere",
    "ispezione",
    "cugino", "cugina",
    "coppia",
    "rivista",
    "cuoco", "cucinare",
    "dipendente",
    "sicurezza",
    "pollo",
    "versione",
    "valuta", "moneta",
    "scala a pioli",
    "schema", "piano",
    "cucina",
    "impiego", "occupazione",
    "locale",
    "attenzione",
    "manager", "responsabile",
    "fatto",
    "coprire", "copertina",
    "triste",
    "guardia", "sorvegliare",
    "parente", "relativo",
    "contea", "provincia",
    "tasso", "tariffa",
    "pranzo",
    "programma",
    "iniziativa",
    "ingranaggio", "attrezzatura",
    "ponte",
    "seno", "petto",
    "parlare", "discorso",
    "piatto vivanda",
    "garanzia",
    "birra",
    "veicolo",
    "ricevimento", "accoglienza",
    "donna",
    "sostanza", "materia",
    "copia", "copiare",
    "lezione", "conferenza",
    "vantaggio",
    "parco",
    "freddo",
    "morte",
    "mescolare",
    "tenere", "prendere",
    "scala", "bilancia",
    "domani",
    "sangue",
    "richiesta",
    "verde",
    "biscotto",
    "chiesa",
    "strisciare", "spogliarello",
    "per sempre",
    "oltre",
    "debito",
    "placcaggio", "affrontare",
    "lavare",
    "seguente",
    "sentire", "provare",
    "massimo",
    "settore",
    "mare",
    "proprietà",
    "economia",
    "menu",
    "panchina", "tribunale",
    "provare", "tentare",
    "lingua",
    "iniziare",
    "chiamata", "telefonare",
    "solido",
    "indirizzo",
    "reddito",
    "piede",
    "senior", "anziano",
    "miele",
    "pochi",
    "miscela",
    "contanti",
    "drogheria", "alimentari",
    "collegamento", "link",
    "mappa",
    "modulo", "forma",
    "fattore",
    "pentola",
    "modello",
    "scrittore", "autore",
    "fattoria",
    "inverno",
    "abilità",
    "ovunque",
    "compleanno",
    "politica",
    "rilascio", "pubblicare",
    "marito",
    "laboratorio",
    "fretta",
    "posta",
    "attrezzatura",
    "lavandino", "affondare",
    "paio", "coppia",
    "autista",
    "considerazione",
    "pelle", "cuoio",
    "pelle", "cute",
    "blu",
    "barca",
    "vendita",
    "mattone",
    "due",
    "nutrire", "alimentare",
    "quadrato", "piazza",
    "punto", "puntino",
    "corsa", "fretta",
    "sogno",
    "posizione", "luogo",
    "pomeriggio",
    "produttore",
    "controllo",
    "occasione",
    "problema", "guaio",
    "introduzione",
    "consiglio",
    "scommessa",
    "mangiare",
    "uccidere",
    "categoria",
    "modo", "maniera",
    "ufficio",
    "proprietà", "tenuta",
    "orgoglio",
    "consapevolezza",
    "scivolare", "slip",
    "crepa", "fessura",
    "cliente",
    "unghia", "chiodo",
    "sparare", "scattare",
    "iscrizione", "abbonamento",
    "morbido", "soffice",
    "chiunque",
    "rete", "web",
    "ufficiale",
    "individuo",
    "pizza",
    "interesse",
    "borsa", "sacca",
    "incantesimo", "scrivere",
    "professione",
    "regina",
    "accordo", "affare",
    "risorsa",
    "nave", "spedire",
    "tipo", "tizio",
    "cioccolato",
    "articolazione", "congiunto",
    "formale",
    "di sopra", "piano superiore",
    "auto", "macchina",
    "resort", "villaggio turistico",
    "all’estero",
    "rivenditore", "dealer",
    "associare", "collega",
    "dito",
    "chirurgia", "operazione",
    "commento",
    "squadra", "team",
    "dettaglio",
    "pazzo",
    "sentiero", "percorso",
    "racconto", "fiaba",
    "iniziale",
    "braccio",
    "radio",
    "domanda", "richiesta",
    "singolo", "celibe", "singola",

    "disegnare", "pareggio", "attrarre", "estrarre",
    "giallo",
    "gara", "concorso", "contestare",
    "pezzo", "brano",
    "citazione", "quotare", "preventivo",
    "tirare", "trazione",
    "commerciale", "spot pubblicitario",
    "camicia", "maglietta",
    "contributo", "apporto",
    "crema", "panna",
    "canale", "canalizzare",
    "abito", "completo", "causa legale", "stare bene a",
    "disciplina", "disciplinare",
    "istruzione", "direttiva",
    "concerto", "intesa",
    "discorso", "parola",
    "basso",
    "efficace", "in vigore",
    "appendere", "impiccare", "restare in linea",
    "graffiare", "graffio", "grattare",
    "industria", "settore",
    "colazione",
    "posare", "sdraiare", "deporre", "laico",
    "unirsi", "giuntura", "congiungere",
    "metallo", "metal",
    "camera da letto",
    "minuto", "minuzioso",
    "prodotto",
    "riposo", "resto", "riposare",
    "temperatura",
    "molti", "tanti",
    "dare", "regalare", "cedere",
    "argomentazione", "lite", "discussione",
    "stampare", "stampa", "impronta",
    "viola", "porpora",
    "ridere", "risata",
    "salute",
    "credito", "riconoscimento",
    "investimento",
    "vendere", "svendere",
    "ambientazione", "impostazione", "scenario", "contesto",
    "lezione", "insegnamento",
    "uovo",
    "medio", "mezzo", "centrale",
    "matrimonio", "nozze",
    "livello",
    "prova", "evidenza", "indizio",
    "frase", "locuzione", "espressione",
    "amore", "amare",
    "sé", "se stesso", "auto-",
    "beneficio", "vantaggio", "benefit",
    "guida", "orientamento",
    "influenzare", "colpire",
    "tu", "voi", "Lei",
    "papà", "babbo", "padre",
    "ansia", "agitazione",
    "speciale", "particolare",
    "fidanzato", "ragazzo",
    "test", "esame", "prova",
    "vuoto", "in bianco",
    "pagamento",
    "zuppa", "minestra", "brodo",
    "obbligo",
    "rispondere", "risposta",
    "sorriso", "sorridere",
    "profondo",
    "lamentela", "reclamo", "querela",
    "aggiunta", "addizione",
    "recensione", "revisione", "rivedere",
    "scatola", "riquadro", "ring",
    "asciugamano", "salvietta",
    "minore", "secondario", "minorenne",
    "divertimento", "divertente",
    "suolo", "terreno", "sporcare",
    "questione", "problema", "emissione", "pubblicare",
    "sigaretta",
    "internet", "rete",
    "guadagno", "ottenere", "aumentare",
    "dire", "raccontare",
    "ingresso", "voce", "iscrizione",
    "di scorta", "risparmiare",
    "incidente", "episodio",
    "famiglia",
    "rifiutare", "rifiuti",
    "filiale", "ramo", "branca",
    "lattina", "barattolo", "potere", "inscatolare",
    "penna", "recinto",
    "nonno",
    "costante",
    "serbatoio", "carro armato", "vasca",
    "zio",
    "clima",
    "terra", "suolo", "fondare",
    "volume",
    "comunicazione",
    "gentile", "tipo",
    "poeta", "poetessa",
    "bambino", "figlio",
    "schermo", "selezionare",
    "mio", "miniera", "mina",
    "smettere", "abbandonare", "licenziarsi",
    "gene",
    "mancanza", "scarsità", "mancare",
    "carità", "beneficenza", "ente benefico",
    "memoria", "ricordo",
    "dente",
    "paura", "temere",
    "menzionare", "menzione", "citare",
    "marketing",
    "rivelare", "svelare",
    "ragione", "motivo", "ragionare",
    "tribunale", "corte", "campo",
    "stagione", "condire",
    "libertà",
    "terra", "atterrare", "sbarcare", "terreno",
    "sport",
    "pubblico", "platea",
    "aula",
    "legge", "diritto",
    "gancio", "agganciare",
    "vincere", "vittoria",
    "portare", "trasportare",
    "occhio",
    "odore", "annusare", "puzzare",
    "distribuzione",
    "ricerca", "fare ricerca",
    "paese", "campagna",
    "osare", "ardire",
    "speranza", "sperare",
    "mentre", "considerato che",
    "allungare", "tratto",
    "biblioteca",
    "se",
    "ritardo", "ritardare",
    "università", "college",
    "plastica", "plastico",
    "libro", "prenotare",
    "presente", "regalo", "presentare",
    "uso", "usare", "utilizzo",
    "preoccuparsi", "preoccupazione",
    "campione", "sostenere",
    "obiettivo", "rete",
    "economia", "risparmio",
    "marzo", "marcia", "marciare",
    "elezione",
    "riflessione", "riflesso",
    "mezzanotte",
    "scivolare", "diapositiva", "scivolo",
    "inflazione", "gonfiaggio",
    "azione", "causa",
    "sfida", "contestare",
    "chitarra",
    "costa",
    "mela",
    "campagna",
    "campo", "settore",
    "giacca", "giubbotto",
    "senso", "percepire",
    "modo", "via", "strada", "maniera",
    "visivo", "visuale",
    "rimuovere", "togliere",
    "meteo", "tempo atmosferico",
    "spazzatura", "buttare",
    "cavo", "tv via cavo",
    "rimpianto", "pentirsi",
    "amico", "compare",
    "spiaggia",
    "storico", "storica",
    "coraggio",
    "simpatia", "compassione",
    "camion",
    "tensione",
    "permesso", "permettere",
    "naso",
    "letto", "aiuola",
    "figlio",
    "persona",
    "base", "basare",
    "carne",
    "solito", "abituale",
    "aria", "aspetto",
    "riunione", "incontro",
    "valore", "vale la pena",
    "gioco", "partita", "selvaggina",
    "indipendenza",
    "fisico", "visita medica",
    "breve", "memoria",
    "giocare", "recitare", "opera teatrale",
    "aumentare", "sollevare", "aumento",
    "tavola", "consiglio", "imbarcare",
    "lei",
    "chiave", "tasto", "fondamentale",
    "scrittura",
    "scegliere", "raccogliere", "piccone",
    "comando", "ordinare",
    "festa", "partito", "parte",
    "ieri",
    "primavera", "molla", "sorgente", "saltare",
    "candidato", "candidata",
    "fisica",
    "università",
    "preoccupazione", "interessare", "impresa",
    "sviluppo",
    "cambiamento", "cambiare", "spiccioli",
    "corda", "filo", "stringa",
    "bersaglio", "obiettivo", "prendere di mira",
    "istanza", "esempio",
    "stanza", "spazio",
    "amaro", "acre", "aspro",
    "uccello",
    "calcio", "football americano",
    "normale",
    "dividere", "spaccare", "divisione",
    "impressione", "impronta",
    "legno", "bosco",
    "lungo", "bramare",
    "significato", "senso",
    "scorta", "azione", "magazzino",
    "cappellino", "tappo", "tetto",
    "leadership", "dirigenza", "capacità di guida",
    "media", "mezzi di comunicazione",
    "ambizione",
    "pesca",
    "tema", "saggio",
    "insalata",
    "riparare", "riparazione",
    "oggi",
    "designer", "stilista", "progettista",
    "notte", "serata",
    "banca", "argine", "sponda",
    "disegno", "estrazione",
    "inevitabile",
    "fase",
    "vasto", "immenso",
    "patatina", "chip", "scheggia", "gettone",
    "rabbia", "ira",
    "interruttore", "scambiare", "cambiare",
    "piangere", "grido",
    "torcere", "colpo di scena", "torsione",
    "personalità",
    "tentativo", "tentare",
    "archiviazione", "magazzino", "stoccaggio",
    "essere", "creatura",
    "preparazione",
    "pipistrello", "mazza", "colpire",
    "selezione", "scelta",
    "bianco",
    "tecnologia",
    "contratto", "contrarre",
    "lato", "fianco", "prendere le parti di",
    "sezione", "tratto",
    "stazione", "postazione", "installare",
    "fino a", "cassa", "lavorare il terreno",
    "struttura", "strutturare",
    "lingua",
    "gusto", "assaggiare",
    "verità",
    "difficoltà",
    "gruppo",
    "limite", "limitare",
    "principale",
    "muovere", "mossa", "traslocare",
    "sensazione", "sentimento",
    "luce", "leggero", "accendere",
    "esempio",
    "missione", "mandato",
    "potere", "forza", "potrebbe",
    "aspettare", "attendere",
    "ruota", "volante",
    "negozio", "fare acquisti",
    "ospite", "presentatore", "host",
    "classico",
    "alternativa", "alternativo",
    "causa", "provocare",
    "agente", "rappresentante",
    "consistere",
    "tavolo", "tabella",
    "compagnia aerea",
    "testo", "inviare un messaggio",
    "piscina", "biliardo", "mettere in comune",
    "artigianato", "mestiere", "imbarcazione",
    "gamma", "intervallo", "catena montuosa", "raggio d'azione",
    "carburante", "alimentare",
    "attrezzo", "strumento",
    "partner", "socio", "compagno",
    "caricare", "carico",
    "ingresso", "entrata",
    "deposito", "acconto", "versare",
    "odiare", "odio",
    "articolo",
    "video",
    "estate",
    "caratteristica", "funzione", "articolo di approfondimento",
    "estremo",
    "mobile", "cellulare",
    "ospedale",
    "volo", "fuga",
    "cadere", "autunno",
    "pensione",
    "pianoforte", "piano",
    "fallire", "bocciarsi",
    "risultato", "risultare",
    "strofinare", "sfregare",
    "divario", "fessura", "lacuna",
    "sistema", "impianto",
    "rapporto", "relazione", "segnalare",
    "succhiare", "fare schifo",
    "ordinario", "comune",
    "vento", "avvolgere",
    "nervo", "coraggio",
    "chiedere", "domandare",
    "splendere", "lucidare", "brillare",
    "nota", "notare", "biglietto",
    "linea", "fila", "riga",
    "mamma", "madre",
    "percezione",
    "fratello",
    "riferimento", "referenza", "citazione",
    "piegare", "curva", "ansa",
    "caricare", "accusa", "costo", "incaricare",
    "trattare", "cura", "leccornia",
    "trucco", "ingannare",
    "termine", "semestre", "condizione",
    "compiti", "compiti a casa",
    "cuocere al forno", "infornare",
    "offerta", "offrire",
    "stato", "status",
    "progetto", "proiettare",
    "strategia",
    "arancione", "arancia",
    "lasciare", "affittare",
    "entusiasmo",
    "genitore",
    "concentrarsi", "concentrato",
    "dispositivo", "congegno",
    "viaggiare", "viaggio",
    "poesia",
    "business", "affari", "azienda",
    "società",
    "bacio", "baciare",
    "fine", "terminare",
    "verdura", "vegetale",
    "assumere", "impiegare",
    "programma", "orario", "pianificare",
    "ora",
    "coraggioso",
    "mettere a fuoco", "concentrarsi", "fuoco",
    "processo", "procedimento", "elaborare",
    "film", "pellicola",
    "illegale",
    "generale", "generico",
    "caffè",
    "annuncio", "pubblicità",
    "autostrada", "superstrada",
    "chimica", "alchimia",
    "psicologia",
    "assumere", "noleggiare",
    "campana", "campanello",
    "conferenza", "convegno",
    "sollievo", "rilievo",
    "spettacolo", "mostrare",
    "ordinato", "figo", "liscio",
    "divertente", "strano",
    "peso",
    "qualità",
    "club", "circolo", "mazza",
    "figlia",
    "zona",
    "toccare", "tocco",
    "stasera", "questa notte",
    "shock", "scuotere", "scioccare",
    "bruciare", "ustione",
    "scusa", "giustificazione", "scusare",
    "nome", "nominare",
    "sondaggio", "rilievo", "ispezionare",
    "paesaggio",
    "anticipo", "avanzare",
    "soddisfazione",
    "pane",
    "disastro", "catastrofe",
    "voce", "articolo", "elemento",
    "cappello",
    "precedente", "priore",
    "shopping", "spesa",
    "visitare", "visita",
    "est", "oriente",
    "foto", "fotografia",
    "casa", "abitazione", "in casa",
    "idea",
    "padre", "babbo",
    "confronto", "comparazione",
    "gatto",
    "tubo", "pipa", "condotta",
    "vincitore", "vincitrice",
    "contare", "conte",
    "lago",
    "combattere", "lotta", "rissa",
    "premio",
    "fondazione", "fondamenta", "fondotinta",
    "cane",
    "tenere", "mantenere", "proseguire",
    "ideale",
    "ventilatore", "tifoso", "fan",
    "lotta", "lottare",
    "vetta", "picco", "ora di punta",
    "sicurezza", "salvaguardia",
    "soluzione",
    "inferno", "diavolo!",
    "conclusione", "esito",
    "popolazione",
    "sforzo", "ceppo", "tensione",
    "allarme", "sveglia",
    "misurazione", "misura",
    "secondo", "secondare",
    "treno", "allenare", "addestrare",
    "gara", "razza", "corsa",
    "dovuto", "scadenza", "previsto",
    "assicurazione",
    "capo", "boss",
    "albero",
    "monitor", "sorvegliare",
    "malato", "stufo",
    "corso", "percorso", "certo",
    "trascinare", "seccatura",
    "appuntamento", "nomina",
    "fetta", "affettare",
    "ancora", "immobile", "alambicco",
    "cura", "preoccuparsi", "interessarsi",
    "pazienza",
    "ricco",
    "fuga", "scappare",
    "emozione",
    "reale", "regale",
    "femmina", "femminile",
    "infanzia",
    "governo", "amministrazione",
    "immagine", "foto", "quadro",
    "volontà", "testamento",
    "calzino", "calza",
    "grande",
    "cancello", "gate",
    "olio", "petrolio",
    "croce", "attraversare", "incrociare", "arrabbiato",
    "spillo", "perno", "PIN",
    "miglioramento",
    "campionato", "titolo",
    "sciocco", "stupido",
    "aiuto", "aiutare",
    "cielo",
    "campo", "intonazione", "lancio", "pece",
    "uomo",
    "diamante", "rombo",
    "la maggior parte", "più",
    "transizione", "passaggio",
    "lavoro", "opera", "lavorare",
    "scienza",
    "comitato", "commissione",
    "momento", "istante",
    "riparare", "fissare", "aggiustare", "imbroglio",
    "insegnamento", "didattica",
    "scavare", "frecciatina",
    "specialista",
    "complesso", "complessato",
    "guida", "guidare",
    "persone", "popolo", "gente",
    "morto", "spento",
    "voce",
    "originale",
    "pausa", "rompere", "interruzione",
    "argomento", "tema",
    "dati",
    "grado", "laurea",
    "lettura", "valore letto",
    "registrazione", "incisione",
    "mucchio", "mazzo", "gruppo",
    "raggiungere", "portata",
    "giudizio", "sentenza",
    "bugia", "mentire", "sdraiarsi",
    "regolare", "abituale",
    "insieme", "set", "ambientazione", "fissare",
    "dipinto", "pittura",
    "modalità", "modo",
    "elenco", "lista",
    "giocatore", "lettore",
    "orso", "sopportare", "partorire",
    "nord", "settentrione",
    "chiedersi", "meraviglia",
    "tappeto", "moquette",
    "pesante", "intenso",
    "ufficiale", "funzionario", "agente",
    "negativo",
    "orologio",
    "unico",
    "bambino", "bebè", "neonato",
    "dolore", "pena",
    "assunzione", "supposizione",
    "disco",
    "ferro", "ferro da stiro", "stirare",
    "conto", "fattura", "banconota", "bolletta", "disegno di legge",
    "cassetto", "disegnatore", "traente",
    "guardare", "sguardo", "aspetto",
    "doppio", "raddoppiare",
    "errore", "sbaglio",
    "finire", "fine", "rifinitura",
    "futuro", "avvenire",
    "brillante", "geniale", "luminoso",
    "contatto", "contattare",
    "matematica",
    "riso",
    "lasciare", "partire", "congedo",
    "ristorante",
    "sconto", "scontare",
    "sesso", "rapporti sessuali",
    "virus",
    "briciola", "po'", "bit",
    "fiducia", "fidarsi", "trust (istituto giuridico)",
    "evento", "avvenimento",
    "indossare", "usura", "consumo",
    "succo", "spremuta",
    "fallimento", "guasto", "mancanza",
    "insetto", "bug (errore)", "difetto",
    "contesto",
    "fango",
    "intero", "tutto", "completo",
    "avvolgere", "involtino", "wrap (panino)",
    "intenzione", "proposito",
    "bozza", "leva (militare)", "corrente d'aria",
    "pressione",
    "torta", "dolce",
    "scuro", "buio", "oscuro",
    "spiegazione",
    "spazio", "cosmo",
    "angolo", "angolazione",
    "parola", "promessa",
    "efficienza", "rendimento",
    "gestione", "direzione", "management",
    "abitudine", "abito",
    "stella", "diva",
    "possibilità", "caso", "occasione",
    "ritrovamento", "constatazione", "risultato",
    "trasporto", "trasporti",
    "stare in piedi", "sopportare", "stand (espositore)", "bancarella",
    "critica", "censura",
    "flusso", "scorrere",
    "porta", "portello",
    "lesione", "infortunio", "ferita",
    "insetto",
    "sorpresa", "sorprendere",
    "appartamento",
]

# WORD_LIST = list(set(WORD_LIST_EN + WORD_LIST_IT))
WORD_LIST = WORD_LIST_IT

# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict(
    {
        "en": "English",
        "es": "Spanish",
        "pt": "Portuguese",
        "ar": "Arabic",
        "hi": "Hindi",
        "fr": "French",
        "ru": "Russian",
        "de": "German",
        "ja": "Japanese",
        "it": "Italian",
        "bn": "Bengali",
        "uk": "Ukrainian",
        "th": "Thai",
        "ur": "Urdu",
        "ta": "Tamil",
        "te": "Telugu",
        "bg": "Bulgarian",
        "ko": "Korean",
        "pl": "Polish",
        "he": "Hebrew",
        "fa": "Persian",
        "vi": "Vietnamese",
        "ne": "Nepali",
        "sw": "Swahili",
        "kn": "Kannada",
        "mr": "Marathi",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "ml": "Malayalam",
        "fi": "Finnish",
    }
)

# _ALPHABETS = "([A-Za-z])"
# _PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
# _SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
# _STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
# _ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
# _WEBSITES = "[.](com|net|org|io|gov|edu|me)"
# _DIGITS = "([0-9])"
# _MULTIPLE_DOTS = r"\.{2,}"

# ITALIAN
_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Sig.|Sig.ra|Sig.na|Sigg|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Sig.|Sig.ra|Sig.na|Sigg.|Dr|Prof||Lui|Lei|Esso|Loro|Nostro|Noi|Ma|Comunque|Questo|Quello|Comunque)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text):
    """Split the text into sentences.

    Args:
      text: A string that consists of more than or equal to one sentences.

    Returns:
      A list of strings where each string is a sentence.
    """
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(_PREFIXES, "\\1<prd>", text)
    text = re.sub(_WEBSITES, "<prd>\\1", text)
    text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
    text = re.sub(
        _MULTIPLE_DOTS,
        lambda match: "<prd>" * len(match.group(0)) + "<stop>",
        text,
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
    text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
    text = re.sub(
        _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(_ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
    text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences


def count_words(text):
    """Counts the number of words."""
    tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)
    num_words = len(tokens)
    return num_words


@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
    return nltk.data.load("nltk:tokenizers/punkt/italian.pickle")


def count_sentences(text):
    """Count the number of sentences."""
    tokenizer = _get_sentence_tokenizer()
    tokenized_sentences = tokenizer.tokenize(text)
    return len(tokenized_sentences)


def generate_keywords(num_keywords):
    """Randomly generates a few keywords."""
    return random.sample(WORD_LIST, k=num_keywords)


##### INSTRUCTIONS.PY

# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# WARN: pip install -e ".[math,ifeval,sentencepiece]"

"""Library of instructions."""

import collections
import json
import logging
import random
import re
import string
from typing import Dict, Optional, Sequence, Union

import langdetect

logger = logging.getLogger(__name__)

_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = LANGUAGE_CODES

# The relational operation for comparison.
# _COMPARISON_RELATION = ("less than", "at least")
_COMPARISON_RELATION = ("meno di", "almeno")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
# _CONSTRAINED_RESPONSE_OPTIONS = (
#     "My answer is yes.",
#     "My answer is no.",
#     "My answer is maybe.",
# )
_CONSTRAINED_RESPONSE_OPTIONS = (
    "La mia risposta è sì.",
    "La mia risposta è no.",
    "La mia risposta è forse.",
)

# The options of starter keywords.
# _STARTER_OPTIONS = (
#     "I would say",
#     "My answer is",
#     "I believe",
#     "In my opinion",
#     "I think",
#     "I reckon",
#     "I feel",
#     "From my perspective",
#     "As I see it",
#     "According to me",
#     "As far as I'm concerned",
#     "To my understanding",
#     "In my view",
#     "My take on it is",
#     "As per my perception",
# )
_STARTER_OPTIONS = (
    "Direi che",
    "La mia risposta è",
    "Credo che",
    "Secondo me",
    "Penso che",
    "Ritengo che",
    "A mio avviso",
    "Dal mio punto di vista",
    "Come la vedo io",
    "A mio parere",
    "Per quanto mi riguarda",
    "Secondo la mia comprensione",
    "Nella mia opinione",
    "La mia valutazione è",
    "Secondo la mia percezione",
)

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
# _ENDING_OPTIONS = ("Any other questions?", "Is there anything else I can help with?")
_ENDING_OPTIONS = ("Hai altre domande?", "C'è altro con cui posso aiutarti?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
# _SECTION_SPLITER = ("Section", "SECTION")
_SECTION_SPLITER = ("Sezione", "SEZIONE", "PARAGRAFO")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
# _POSTSCRIPT_MARKER = ("P.S.", "P.P.S")
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S", "N.B.")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The occurrences of words with all capital letters.
_ALL_CAPITAL_WORD_FREQUENCY = 20

# The number of words in the response.
_NUM_WORDS_LOWER_LIMIT = 100
_NUM_WORDS_UPPER_LIMIT = 500


class Instruction:
    """An instruction template."""

    def __init__(self, instruction_id):
        self.id = instruction_id

    def build_description(self, **kwargs):
        raise NotImplementedError("`build_description` not implemented.")

    def get_instruction_args(self):
        raise NotImplementedError("`get_instruction_args` not implemented.")

    def get_instruction_args_keys(self):
        raise NotImplementedError("`get_instruction_args_keys` not implemented.")

    def check_following(self, value):
        raise NotImplementedError("`check_following` not implemented.")


class ResponseLanguageChecker(Instruction):
    """Check the language of the entire response."""

    def build_description(self, *, language=None):
        """Build the instruction description.

        Args:
          language: A string representing the expected language of the response. The
            language has to comply to the 97 types defined in
            `langid.py` (https://pypi.org/project/langid/1.1.5/), which follows
            ISO 639-1 codes (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes);
            for example, `en` for English, `zh` for Chinese, `fr` for French.

        Returns:
          A string representing the instruction description.
        """
        self._language = language
        if self._language is None:
            self._language = random.choice(list(_LANGUAGES.keys()))
        # TODO(tianjianlu): opens the description generation to more choices.
        self._description_pattern = (
            # "Your ENTIRE response should be in {language} language, no other language is allowed."
            "La tua INTERA risposta deve essere nella lingua {language}, nessun'altra lingua è permessa."

        )
        return self._description_pattern.format(language=_LANGUAGES[self._language])

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"language": self._language}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["language"]

    def check_following(self, value):
        """Check if the language of the entire response follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the language of `value` follows instruction; otherwise False.
        """
        assert isinstance(value, str)

        try:
            return langdetect.detect(value) == self._language
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class NumberOfSentences(Instruction):
    """Check the number of sentences."""

    def build_description(self, *, num_sentences=None, relation=None):
        """Build the instruction description.

        Args:
          num_sentences: An integer specifying the number of sentences as a
            threshold.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of sentences < the threshold;
            if 'at least', the actual number of sentences >= the threshold.

        Returns:
          A string representing the instruction description.
        """
        # The number of sentences as a threshold for comparison.
        self._num_sentences_threshold = num_sentences
        if self._num_sentences_threshold is None or self._num_sentences_threshold < 0:
            self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                # f"{_COMPARISON_RELATION}, but {relation} is given."
                f"{_COMPARISON_RELATION}, ma {relation} è dato."
            )
        else:
            self._comparison_relation = relation

        self._description_pattern = (
            # "Your response should contain {relation} {num_sentences} sentences."
            "La tua risposta deve contenere {relation} {num_sentences} frasi."

        )
        return self._description_pattern.format(
            relation=self._comparison_relation,
            num_sentences=self._num_sentences_threshold,
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "relation"]

    def check_following(self, value):
        """Check if the number of sentences follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the response follows the instruction.

        Raise:
            ValueError if the string in `instruction_args` is not in
            [`less_than`, `at_least`].
        """
        num_sentences = count_sentences(value)
        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_sentences < self._num_sentences_threshold
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
    """Check the placeholders in template writing."""

    def build_description(self, *, num_placeholders=None):
        """Build the instruction description.

        Args:
          num_placeholders: An integer denoting the minimum number of
            placeholders required in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_placeholders = num_placeholders
        if self._num_placeholders is None or self._num_placeholders < 0:
            self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
        # self._description_pattern = (
        #     "The response must contain at least {num_placeholders} placeholders "
        #     + "represented by square brackets, such as [address]."
        # )
        self._description_pattern = (
                "La tua risposta deve contenere almeno {num_placeholders} segnaposti "
                + "rappresentati da parentesi quadre, come [indirizzo]."
        )
        return self._description_pattern.format(num_placeholders=self._num_placeholders)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_placeholders": self._num_placeholders}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_placeholders"]

    def check_following(self, value):
        """Check if the number of placeholders follows the instruction.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual number of placeholders in the response is greater than
          or equal to `num_placeholders`; otherwise, False.
        """
        placeholders = re.findall(r"\[.*?\]", value)
        num_placeholders = len(placeholders)
        return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
    """Checks the bullet list in the prompt."""

    def build_description(self, *, num_bullets=None):
        """Build the instruction description.

        Args:
          num_bullets: An integer specifying the exact number of bullet lists
            that is required to appear in the response.

        Returns:
          A string representing the instruction description.
        """
        self._num_bullets = num_bullets
        if self._num_bullets is None or self._num_bullets < 0:
            self._num_bullets = random.randint(1, _NUM_BULLETS)
        self._description_pattern = (
            # "Your answer must contain exactly {num_bullets} bullet points. "
            # + "Use the markdown bullet points such as:\n"
            # + "* This is point 1. \n"
            # + "* This is point 2"
                "La tua risposta deve contenere esattamente {num_bullets} punti elenco. "
                + "Usa il markdown ad elenco, per esempio:\n"
                + "* Punto 1\n"
                + "* Punto 2"
        )
        return self._description_pattern.format(num_bullets=self._num_bullets)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_bullets": self._num_bullets}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_bullets"]

    def check_following(self, value):
        r"""Check if the number of bullet lists meets the requirement.

        Args:
          value: A string representing the response. The response is expected to
            contain some bullet lists that start with `\*`.

        Returns:
          True if the actual number of bullet lists in the response meets the
          requirement.
        """
        bullet_lists = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
        bullet_lists_2 = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        num_bullet_lists = len(bullet_lists) + len(bullet_lists_2)
        return num_bullet_lists == self._num_bullets


class ConstrainedResponseChecker(Instruction):
    """Checks the constrained response."""

    def build_description(self):
        """Build the instruction description."""
        # A sequence of string(s) representing the options of the expected response.
        self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
        self._description_pattern = (
            # "Answer with one of the following options: {response_options}"
            "Rispondi con una delle seguenti opzioni: {response_options}"
        )
        return self._description_pattern.format(
            response_options=self._constrained_responses
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response matches the constrained options.

        Args:
          value: A string representing the response.

        Returns:
          True if the actual response contains one of the options in the constrained
          responses; otherwise False.
        """
        value = value.strip()
        for constrained_response in self._constrained_responses:
            if constrained_response in value:
                return True
        return False


class ConstrainedStartChecker(Instruction):
    """Checks the response start."""

    def build_description(self, *, starter=None):
        """Build the instruction description.

        Args:
          starter: A string representing the keyward that the response should start
            with.

        Returns:
          A string representing the instruction description.
        """
        self._starter = starter.strip() if isinstance(starter, str) else starter
        if self._starter is None:
            self._starter = random.choice(_STARTER_OPTIONS)
        self._description_pattern = (
            # "During the conversation, when it is your turn, please always start with {starter}"
            "Durante la conversazione, quando tocca a te, inizia sempre con {starter}"
        )
        return self._description_pattern.format(starter=self._starter)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"starter": self._starter}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["starter"]

    def check_following(self, value):
        """Checks if the response starts with the constrained keyword or phrase.

        Args:
          value: A string representing the response.

        Returns:
          True if the response starts with the given phrase or keyword that is
          contained in `instruction_args`; otherwise, False.
        """
        response_pattern = r"^\s*" + self._starter + r".*$"
        response_with_constrained_start = re.search(
            response_pattern, value, flags=re.MULTILINE
        )
        return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
    """Checks the highlighted section."""

    def build_description(self, *, num_highlights=None):
        """Build the instruction description.

        Args:
          num_highlights: An integer specifying the minimum number of highlighted
            sections.

        Returns:
          A string representing the instruction description.
        """
        self._num_highlights = num_highlights
        if self._num_highlights is None or self._num_highlights < 0:
            self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

        self._description_pattern = (
            # "Highlight at least {num_highlights} sections in your answer with markdown, i.e. *highlighted section*."
            "Evidenzia almeno {num_highlights} sezioni nella tua risposta con markdown, ad es. *sezione evidenziata*."
        )

        return self._description_pattern.format(num_highlights=self._num_highlights)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_highlights": self._num_highlights}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_highlights"]

    def check_following(self, value):
        """Checks if the number of highlighted sections meets the requirement.

        Args:
          value: a string repesenting the response. The response is expected to
            contain highlighted sections in the format of *highlighted*.

        Returns:
          True if the actual number of highlighted sections in the format of
          *highlighed sections* meets the minimum requirement; otherwise False.
        """
        num_highlights = 0
        highlights = re.findall(r"\*[^\n\*]*\*", value)
        double_highlights = re.findall(r"\*\*[^\n\*]*\*\*", value)
        for highlight in highlights:
            if highlight.strip("*").strip():
                num_highlights += 1
        for highlight in double_highlights:
            if highlight.removeprefix("**").removesuffix("**").strip():
                num_highlights += 1

        return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
    """Checks the sections."""

    def build_description(self, *, section_spliter=None, num_sections=None):
        """Build the instruction description.

        Args:
          section_spliter: A string represents the section spliter keyword that
            marks a new section, i.e., `Section` or `SECTION`.
          num_sections: An integer specifying the number of sections.

        Returns:
          A string representing the instruction description.
        """
        self._section_spliter = (
            section_spliter.strip()
            if isinstance(section_spliter, str)
            else section_spliter
        )
        if self._section_spliter is None:
            self._section_spliter = random.choice(_SECTION_SPLITER)

        self._num_sections = num_sections
        if self._num_sections is None or self._num_sections < 0:
            self._num_sections = random.randint(1, _NUM_SECTIONS)

        self._description_pattern = (
            # "Your response must have {num_sections} sections. Mark the beginning "
            # + "of each section with {section_spliter} X, such as:\n"
            # + "{section_spliter} 1\n"
            # + "[content of section 1]\n"
            # + "{section_spliter} 2\n"
            # + "[content of section 2]"
                "La tua risposta deve avere {num_sections} sezioni. Segna l'inizio "
                + "di ogni sezione con {section_spliter} X, come:\n"
                + "{section_spliter} 1\n"
                + "[contenuto della sezione 1]\n"
                + "{section_spliter} 2\n"
                + "[contenuto della sezione 2]"
        )

        return self._description_pattern.format(
            num_sections=self._num_sections, section_spliter=self._section_spliter
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "section_spliter": self._section_spliter,
            "num_sections": self._num_sections,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["section_spliter", "num_sections"]

    def check_following(self, value):
        """Checks the response contains multiple sections.

        Args:
          value: A string representing the response. The response is expected
            to contain multiple sections (number of sections is greater than 1).
            A new section starts with `Section 1`, where the number denotes the
            section index.

        Returns:
          True if the number of sections in the response is greater than or equal to
          the minimum number of sections; otherwise, False.
        """
        section_splitter_patten = r"\s?" + self._section_spliter + r"\s?\d+\s?"
        sections = re.split(section_splitter_patten, value)
        num_sections = len(sections) - 1
        return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
    """Checks the paragraphs."""

    def build_description(self, *, num_paragraphs=None):
        """Build the instruction description.

        Args:
          num_paragraphs: An integer specifying the number of paragraphs.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._description_pattern = (
            # "There should be {num_paragraphs} paragraphs. "
            # + "Paragraphs are separated with the markdown divider: ***"
                "Ci devono essere {num_paragraphs} paragrafi. "
                + "I paragrafi sono separati con il divisore markdown: ***"
        )

        return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_paragraphs": self._num_paragraphs}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs"]

    def check_following(self, value):
        """Checks the response contains required number of paragraphs.

        Args:
          value: A string representing the response. The response may contain
            paragraphs that are separated by the markdown divider: `***`.

        Returns:
          True if the actual number of paragraphs is the same as required;
          otherwise, False.
        """
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        num_paragraphs = len(paragraphs)

        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    num_paragraphs -= 1
                else:
                    return False

        return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
    """Checks the postscript."""

    def build_description(self, *, postscript_marker=None):
        """Build the instruction description.

        Args:
          postscript_marker: A string containing the keyword that marks the start
            of the postscript section.

        Returns:
          A string representing the instruction description.
        """
        self._postscript_marker = (
            postscript_marker.strip()
            if isinstance(postscript_marker, str)
            else postscript_marker
        )
        if self._postscript_marker is None:
            self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

        self._description_pattern = (
            # "At the end of your response, please explicitly add a postscript "
            # + "starting with {postscript}"
                "Alla fine della tua risposta, aggiungi esplicitamente un post scriptum "
                + "che inizi con {postscript}"
        )

        return self._description_pattern.format(postscript=self._postscript_marker)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"postscript_marker": self._postscript_marker}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["postscript_marker"]

    def check_following(self, value):
        """Checks if the response follows the postscript format.

        Args:
          value: a string representing the response. The response is expected to
            contain a postscript section.

        Returns:
          True if the response contains a postscript section starting with
          the keyword containing in the `instruction_args`; otherwise False.
        """
        value = value.lower()
        if self._postscript_marker == "P.P.S":
            postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif self._postscript_marker == "P.S.":
            postscript_pattern = r"\s*p\.\s?s\..*$"
        elif self._postscript_marker == "N.B.":
            postscript_pattern = r"\s*N\.\s?B\..*$"
        else:
            postscript_pattern = r"\s*" + self._postscript_marker.lower() + r".*$"
        postscript = re.findall(postscript_pattern, value, flags=re.MULTILINE)
        return True if postscript else False


class RephraseChecker(Instruction):
    """Checks the repharse."""

    def build_description(self, *, original_message):
        """Build the instruction description.

        Args:
          original_message: A string representing the original message. The
            rephrased response should only change its words/sentences in between
            its two asterisks, for example, *change me*. Both original and rephrased
            messages should contain the changes in the form of *change me*.

        Returns:
          A string representing the instruction description.
        """
        if not self.is_change(original_message):
            raise ValueError(
                f"Message {original_message} does not contain changes "
                "in the form of *change me*."
            )

        self._reference_without_change = original_message
        self._description = (
            # "Rephrasing: Your rephrased response should only"
            # + "change the words/sentences in between two asterisks"
            # + "such as *change me*."
                "Riformulazione: La tua risposta riformulata dovrebbe solo"
                + "cambiare le parole/frasi tra due asterischi"
                + "come *cambiare me*."
        )
        return self._description

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"original_message": self._reference_without_change}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_message"]

    def check_following(self, value):
        r"""Checks if the rephrasing follows the instruction.

        Args:
          value: A string representing the response, which is expected to rephras
            the string of `instruction_args`.

        Returns:
          True if `value` and `instruction_args` only differ by the words/sentences
          in between two asterisks such as *change me*; otherwise, False.
        """

        if not self.is_change(value):
            raise ValueError(
                f"value {value} does not contain " "changes in the form of *change me*."
            )

        response_without_changes = self.strip_changes(value)
        reference_without_changes = self.strip_changes(self._reference_without_change)

        return response_without_changes == reference_without_changes

    def is_change(self, response):
        """Check if there is change in the response in the form of *change me*."""
        return re.search(r"\*.*\*", response)

    def strip_changes(self, response):
        """Strips off the changes."""
        return re.sub(r"\*.*\*", "", response)


class KeywordChecker(Instruction):
    """Check the exisitence of certain keywords."""

    def build_description(self, *, keywords=None):
        """Build the instruction description.

        Args:
          keywords: A sequence of strings representing the keywords that are
            expected in the response.

        Returns:
          A string representing the instruction description.
        """

        if not keywords:
            self._keywords = generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._keywords = keywords
        self._keywords = sorted(self._keywords)

        # self._description_pattern = "Include keywords {keywords} in the response."
        self._description_pattern = "Includi le parole chiave {keywords} nella risposta."

        return self._description_pattern.format(keywords=self._keywords)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"keywords": self._keywords}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keywords"]

    def check_following(self, value):
        """Check if the response contain the expected keywords."""
        for keyword in self._keywords:
            if not re.search(keyword, value, flags=re.IGNORECASE):
                return False
        return True


class KeywordFrequencyChecker(Instruction):
    """Check the keyword frequency."""

    def build_description(self, *, keyword=None, frequency=None, relation=None):
        """Build the instruction description.

        Args:
          keyword: A string representing a keyword that is expected in the response.
          frequency: An integer specifying the number of times `keyword` is expected
            to appear in the response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of occurrences < frequency;
            if 'at least', the actual number of occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if not keyword:
            self._keyword = generate_keywords(num_keywords=1)[0]
        else:
            self._keyword = keyword.strip()

        self._frequency = frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                # f"{_COMPARISON_RELATION}, but {relation} is given."
                f"{_COMPARISON_RELATION}, ma {relation} è dato."
            )
        else:
            self._comparison_relation = relation

        # self._description_pattern = (
        #     "In your response, the word {keyword} should appear {relation} "
        #     + "{frequency} times."
        # )
        self._description_pattern = (
                "Nella tua risposta, la parola {keyword} dovrebbe apparire {relation} "
                + "{frequency} volte."
        )

        return self._description_pattern.format(
            keyword=self._keyword,
            relation=self._comparison_relation,
            frequency=self._frequency,
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["keyword", "frequency", "relation"]

    def check_following(self, value):
        """Checks if the response contain the keyword with required frequency."""
        actual_occurrences = len(re.findall(self._keyword, value, flags=re.IGNORECASE))

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return actual_occurrences < self._frequency
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return actual_occurrences >= self._frequency


class NumberOfWords(Instruction):
    """Checks the number of words."""

    def build_description(self, *, num_words=None, relation=None):
        """Build the instruction description.

        Args:
          num_words: An integer specifying the number of words contained in the
            response.
          relation: A string in (`less than`, `at least`), defining the relational
            operator for comparison.
            Two relational comparisons are supported for now:
            if 'less than', the actual number of words < num_words;
            if 'at least', the actual number of words >= num_words.

        Returns:
          A string representing the instruction description.
        """

        self._num_words = num_words
        if self._num_words is None or self._num_words < 0:
            self._num_words = random.randint(
                _NUM_WORDS_LOWER_LIMIT, _NUM_WORDS_UPPER_LIMIT
            )

        if relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                # f"{_COMPARISON_RELATION}, but {relation} is given."
                f"{_COMPARISON_RELATION}, ma {relation} è dato."            
            )
        else:
            self._comparison_relation = relation

        # self._description_pattern = "Answer with {relation} {num_words} words."
        self._description_pattern = "Rispondi con {relation} {num_words} parole."

        return self._description_pattern.format(
            relation=self._comparison_relation, num_words=self._num_words
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"num_words": self._num_words, "relation": self._comparison_relation}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_words", "relation"]

    def check_following(self, value):
        """Checks if the response contains the expected number of words."""
        num_words = count_words(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return num_words < self._num_words
        elif self._comparison_relation == _COMPARISON_RELATION[1]:
            return num_words >= self._num_words


class JsonFormat(Instruction):
    """Check the Json format."""

    def build_description(self):
        self._description_pattern = (
            # "Entire output should be wrapped in JSON format. You can use markdown"
            # " ticks such as ```."
            "L'intera output deve essere racchiusa in formato JSON. Puoi usare i backtick markdown come ```."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        value = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError:
            return False
        return True


class ParagraphFirstWordCheck(Instruction):
    """Check the paragraph and the first word of the nth paragraph."""

    def build_description(
            self, num_paragraphs=None, nth_paragraph=None, first_word=None
    ):
        r"""Build the instruction description.

        Args:
          num_paragraphs: An integer indicating the number of paragraphs expected
            in the response. A paragraph is a subset of the string that is
            expected to be separated by '\n\n'.
          nth_paragraph: An integer indicating the paragraph number that we look at.
            Note that n starts from 1.
          first_word: A string that represent the first word of the bth paragraph.

        Returns:
          A string representing the instruction description.
        """
        self._num_paragraphs = num_paragraphs
        if self._num_paragraphs is None or self._num_paragraphs < 0:
            self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

        self._nth_paragraph = nth_paragraph
        if (
                self._nth_paragraph is None
                or self._nth_paragraph <= 0
                or self._nth_paragraph > self._num_paragraphs
        ):
            self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

        self._first_word = first_word
        if self._first_word is None:
            self._first_word = generate_keywords(num_keywords=1)[0]
        self._first_word = self._first_word.lower()

        self._description_pattern = (
            # "There should be {num_paragraphs} paragraphs. "
            # + "Paragraphs and only paragraphs are separated with each other by two "
            # + "new lines as if it was '\\n\\n' in python. "
            # + "Paragraph {nth_paragraph} must start with word {first_word}."
                "Ci devono essere {num_paragraphs} paragrafi. "
                + "I paragrafi e solo i paragrafi sono separati l'uno dall'altro da due "
                + "nuove righe come se fosse '\\n\\n' in python. "
                + "Il paragrafo {nth_paragraph} deve iniziare con la parola {first_word}."
        )

        return self._description_pattern.format(
            num_paragraphs=self._num_paragraphs,
            nth_paragraph=self._nth_paragraph,
            first_word=self._first_word,
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_paragraphs", "nth_paragraph", "first_word"]

    def check_following(self, value):
        """Checks for required number of paragraphs and correct first word.

        Args:
          value: a string representing the response. The response may contain
            paragraphs that are separated by two new lines and the first word of
            the nth paragraph will have to match a specified word.

        Returns:
          True if the number of paragraphs is the same as required and the first
          word of the specified paragraph is the same as required. Otherwise, false.
        """

        paragraphs = re.split(r"\n\n", value)
        num_paragraphs = len(paragraphs)

        for paragraph in paragraphs:
            if not paragraph.strip():
                num_paragraphs -= 1

        # check that index doesn't go out of bounds
        if self._nth_paragraph <= num_paragraphs:
            paragraph = paragraphs[self._nth_paragraph - 1].strip()
            if not paragraph:
                return False
        else:
            return False

        first_word = ""
        punctuation = {".", ",", "?", "!", "'", '"'}

        # get first word and remove punctuation
        word = paragraph.split()[0].strip()
        # TODO(jeffrey): make more complex?
        word = word.lstrip("'")
        word = word.lstrip('"')

        for letter in word:
            if letter in punctuation:
                break
            first_word += letter.lower()

        return num_paragraphs == self._num_paragraphs and first_word == self._first_word


# TODO(jeffrey) add relation - at least/at most?
class KeySentenceChecker(Instruction):
    """Check the existence of certain key sentences."""

    def build_description(self, key_sentences=None, num_sentences=None):
        """Build the instruction description.

        Args:
          key_sentences: A sequences of strings representing the key sentences that
            are expected in the response.
          num_sentences: The number of key sentences that are expected to be seen in
            the response.

        Returns:
          A string representing the instruction description.
        """

        if not key_sentences:
            # TODO(jeffrey) make a generate sentences function? wonderwords package
            self._key_sentences = set(["For now, this is fine."])
        else:
            self._key_sentences = key_sentences

        if not num_sentences:
            self._num_sentences = random.randint(1, len(self._key_sentences))
        else:
            self._num_sentences = num_sentences

        self._description_pattern = (
            # "Include {num_sentences} of the following sentences {key_sentences}"
            "Includi {num_sentences} delle seguenti frasi {key_sentences}"
        )

        return self._description_pattern.format(
            num_sentences=self._num_sentences, key_sentences=self._key_sentences
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "num_sentences": self._num_sentences,
            "key_sentences": list(self._key_sentences),
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["num_sentences", "key_sentences"]

    def check_following(self, value):
        """Checks if the response contains the expected key sentences."""
        count = 0
        sentences = split_into_sentences(value)
        for sentence in self._key_sentences:
            if sentence in sentences:
                count += 1

        return count == self._num_sentences


class ForbiddenWords(Instruction):
    """Checks that specified words are not used in response."""

    def build_description(self, forbidden_words=None):
        """Build the instruction description.

        Args:
          forbidden_words: A sequences of strings respresenting words that are not
            allowed in the response.

        Returns:
          A string representing the instruction description.
        """

        if not forbidden_words:
            self._forbidden_words = generate_keywords(
                num_keywords=_NUM_KEYWORDS
            )
        else:
            self._forbidden_words = list(set(forbidden_words))
        self._forbidden_words = sorted(self._forbidden_words)
        self._description_pattern = (
            # "Do not include keywords {forbidden_words} in the response."
            "Non includere le parole chiave {forbidden_words} nella risposta."
        )

        return self._description_pattern.format(forbidden_words=self._forbidden_words)

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {"forbidden_words": self._forbidden_words}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["forbidden_words"]

    def check_following(self, value):
        """Check if the response does not contain the expected keywords."""
        for word in self._forbidden_words:
            if re.search(r"\b" + word + r"\b", value, flags=re.IGNORECASE):
                return False
        return True


class RephraseParagraph(Instruction):
    """Checks that the paragraph is rephrased."""

    def build_description(self, *, original_paragraph, low, high):
        """Builds the instruction description.

        Args:
          original_paragraph: A string presenting the original paragraph. The
            rephrases response should have betweeb low-high words in common.
          low: An integer presenting the lower bound of similar words.
          high: An integer representing the upper bound of similar words.

        Returns:
          A string representing the instruction description.
        """
        # TODO(jeffrey) make more encompassing
        self._original_paragraph = original_paragraph
        self._low = low
        self._high = high

        self._description = (
            # "Rephrase the following paragraph: "
            # + "{original_paragraph}\nYour response should have "
            # + "between {low} and {high} of the same words. "
            # + "Words are the same if and only if all of the "
            # + "letters, ignoring cases, are the same. For "
            # + "example, 'run' is the same as 'Run' but different "
            # + "to 'ran'."
                "Riformula il seguente paragrafo: "
                + "{original_paragraph}\nLa tua risposta dovrebbe avere "
                + "tra {low} e {high} delle stesse parole. "
                + "Le parole sono le stesse se e solo se tutte le "
                + "lettere, ignorando maiuscole e minuscole, sono le stesse. Ad esempio, "
                + "'run' è lo stesso di 'Run' ma diverso da 'ran'."
        )

        return self._description.format(
            original_paragraph=original_paragraph, low=self._low, high=self._high
        )

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return {
            "original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["original_paragraph", "low", "high"]

    def check_following(self, value):
        val_words = re.findall(r"\w+", value.lower())
        original_words = re.findall(r"\w+", self._original_paragraph.lower())
        similar_words = 0

        dict_val = collections.Counter(val_words)
        dict_original = collections.Counter(original_words)

        for word in dict_original:
            similar_words += min(dict_original[word], dict_val[word])

        return similar_words >= self._low and similar_words <= self._high


class TwoResponsesChecker(Instruction):
    """Check that two responses were given."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            # "Give two different responses. Responses and only responses should"
            # " be separated by 6 asterisk symbols: ******."
            "Fornisci due risposte diverse. Le risposte e solo le risposte dovrebbero"
            " essere separate da 6 simboli asterisco: ******."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyward args of `build_description`."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response has two different answers.

        Args:
          value: A string representing the response.

        Returns:
          True if two responses are detected and false otherwise.
        """
        valid_responses = list()
        responses = value.split("******")
        for index, response in enumerate(responses):
            if not response.strip():
                if index != 0 and index != len(responses) - 1:
                    return False
            else:
                valid_responses.append(response)
        return (
                len(valid_responses) == 2
                and valid_responses[0].strip() != valid_responses[1].strip()
        )


class RepeatPromptThenAnswer(Instruction):
    """Checks that Prompt is first repeated then answered."""

    def build_description(self, *, prompt_to_repeat=None):
        """Build the instruction description.

        Args:
          prompt_to_repeat: The prompt that is meant to be repeated.

        Returns:
          A string representing the instruction description.
        """
        if not prompt_to_repeat:
            raise ValueError("prompt_to_repeat must be set.")
        else:
            self._prompt_to_repeat = prompt_to_repeat
        self._description_pattern = (
            # "First repeat the request word for word without change,"
            # " then give your answer (1. do not say any words or characters"
            # " before repeating the request; 2. the request you need to repeat"
            # " does not include this sentence)"
            "Per prima cosa ripeti la richiesta parola per parola senza cambiamenti,"
            " poi dai la tua risposta (1. non dire nessuna parola o carattere"
            " prima di ripetere la richiesta; 2. la richiesta che devi ripetere"
            " non include questa frase)"
        )
        return self._description_pattern

    def get_instruction_args(self):
        return {"prompt_to_repeat": self._prompt_to_repeat}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["prompt_to_repeat"]

    def check_following(self, value):
        if value.strip().lower().startswith(self._prompt_to_repeat.strip().lower()):
            return True
        return False


class EndChecker(Instruction):
    """Checks that the prompt ends with a given phrase."""

    def build_description(self, *, end_phrase=None):
        """Build the instruction description.

        Args:
          end_phrase: A string representing the phrase the response should end with.

        Returns:
          A string representing the instruction description.
        """
        self._end_phrase = (
            end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
        )
        if self._end_phrase is None:
            self._end_phrase = random.choice(_ENDING_OPTIONS)
        self._description_pattern = (
            # "Finish your response with this exact phrase {ender}. "
            # "No other words should follow this phrase."
            "Termina la tua risposta con questa esatta frase {ender}. "
            "Nessun'altra parola dovrebbe seguire questa frase."
        )
        return self._description_pattern.format(ender=self._end_phrase)

    def get_instruction_args(self):
        return {"end_phrase": self._end_phrase}

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["end_phrase"]

    def check_following(self, value):
        """Checks if the response ends with the expected phrase."""
        value = value.strip().strip('"').lower()
        self._end_phrase = self._end_phrase.strip().lower()
        return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
    """Checks the response for a title."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            # "Your answer must contain a title, wrapped in double angular brackets,"
            # " such as <<poem of joy>>."
            "La tua risposta deve contenere un titolo, racchiuso tra doppi simboli di"
            " parentesi angolari, come <<poema di gioia>>."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response contains a title."""
        pattern = r"<<[^\n]+>>"
        re_pattern = re.compile(pattern)
        titles = re.findall(re_pattern, value)

        for title in titles:
            if title.lstrip("<").rstrip(">").strip():
                return True
        return False


class LetterFrequencyChecker(Instruction):
    """Checks letter frequency."""

    def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
        """Build the instruction description.

        Args:
          letter: A string representing a letter that is expected in the response.
          let_frequency: An integer specifying the number of times `keyword` is
            expected to appear in the response.
          let_relation: A string in (`less than`, `at least`), defining the
            relational operator for comparison. Two relational comparisons are
            supported for now; if 'less than', the actual number of
            occurrences < frequency; if 'at least', the actual number of
            occurrences >= frequency.

        Returns:
          A string representing the instruction description.
        """
        if (
                not letter
                or len(letter) > 1
                or ord(letter.lower()) < 97
                or ord(letter.lower()) > 122
        ):
            self._letter = random.choice(list(string.ascii_letters))
        else:
            self._letter = letter.strip()
        self._letter = self._letter.lower()

        self._frequency = let_frequency
        if self._frequency is None or self._frequency < 0:
            self._frequency = random.randint(1, _LETTER_FREQUENCY)

        if let_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif let_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {let_relation} is given."
            )
        else:
            self._comparison_relation = let_relation

        self._description_pattern = (
            # "In your response, the letter {letter} should appear {let_relation}"
            # " {let_frequency} times."
            "Nella tua risposta, la lettera {letter} dovrebbe apparire {let_relation}"
            " {let_frequency} volte."
        )

        return self._description_pattern.format(
            letter=self._letter,
            let_frequency=self._frequency,
            let_relation=self._comparison_relation,
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["letter", "let_frequency", "let_relation"]

    def check_following(self, value):
        """Checks that the response contains the letter at the right frequency."""
        value = value.lower()
        letters = collections.Counter(value)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return letters[self._letter] < self._frequency
        else:
            return letters[self._letter] >= self._frequency


class CapitalLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all capital letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            # "Your entire response should be in English, and in all capital letters."
            "La tua risposta deve essere interamente in italiano e con tutte le lettere maiuscole."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all capital letters."""
        assert isinstance(value, str)

        try:
            return value.isupper() and langdetect.detect(value) == "it"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class LowercaseLettersEnglishChecker(Instruction):
    """Checks that the response is in english and is in all lowercase letters."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            # "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
            "La tua risposta deve essere interamente in italiano, tutta in minuscolo. Nessuna lettera maiuscola è permessa."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response is in English and in all lowercase letters."""
        assert isinstance(value, str)

        try:
            return value.islower() and langdetect.detect(value) == "it"
        except langdetect.LangDetectException as e:
            # Count as instruction is followed.
            logging.error(
                "Unable to detect language for text %s due to %s", value, e
            )  # refex: disable=pytotw.037
            return True


class CommaChecker(Instruction):
    """Checks the response for no commas."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            # "In your entire response, refrain from the use of any commas."
            "In tutta la tua risposta, astieniti dall'uso di virgole."
        )
        return self._description_pattern

    def get_instruction_args(self):
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks that the response does not contain commas."""
        return not re.search(r"\,", value)


class CapitalWordFrequencyChecker(Instruction):
    """Checks frequency of words with all capital letters."""

    def build_description(
            self,
            capital_frequency=None,
            capital_relation=None,
    ):
        """Build the instruction description.

        Args:
          capital_frequency: An integer that represents the number of words that
            should be in all capital letters.
          capital_relation: A string that is 'at least' or 'at most' that refers to
            the frequency.

        Returns:
          A string representing the instruction description.
        """
        self._frequency = capital_frequency
        if self._frequency is None:
            self._frequency = random.randint(1, _ALL_CAPITAL_WORD_FREQUENCY)

        self._comparison_relation = capital_relation
        if capital_relation is None:
            self._comparison_relation = random.choice(_COMPARISON_RELATION)
        elif capital_relation not in _COMPARISON_RELATION:
            raise ValueError(
                "The supported relation for comparison must be in "
                f"{_COMPARISON_RELATION}, but {capital_relation} is given."
            )

        self._description_pattern = (
            # "In your response, words with all capital letters should appear"
            # " {relation} {frequency} times."
            "Nella tua risposta, le parole con tutte le lettere maiuscole dovrebbero apparire"
            " {relation} {frequency} volte."
        )

        return self._description_pattern.format(
            frequency=self._frequency, relation=self._comparison_relation
        )

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return {
            "capital_frequency": self._frequency,
            "capital_relation": self._comparison_relation,
        }

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return ["capital_frequency", "capital_relation"]

    def check_following(self, value):
        """Checks the frequency of words with all capital letters."""
        # Hyphenated words will count as one word
        words = nltk.word_tokenize(value)
        capital_words = [word for word in words if word.isupper()]

        capital_words = len(capital_words)

        if self._comparison_relation == _COMPARISON_RELATION[0]:
            return capital_words < self._frequency
        else:
            return capital_words >= self._frequency


class QuotationChecker(Instruction):
    """Checks response is wrapped with double quotation marks."""

    def build_description(self):
        """Build the instruction description."""
        self._description_pattern = (
            # "Wrap your entire response with double quotation marks."
            "Racchiudi l'intera risposta tra doppi segni di punteggiatura."
        )
        return self._description_pattern

    def get_instruction_args(self):
        """Returns the keyword args of build description."""
        return None

    def get_instruction_args_keys(self):
        """Returns the args keys of `build_description`."""
        return []

    def check_following(self, value):
        """Checks if the response is wrapped with double quotation marks."""
        value = value.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'


##### INSTRUCTION REGISTRY.PY

# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all instructions."""

_KEYWORD = "keywords:"

_LANGUAGE = "language:"

_LENGTH = "length_constraints:"

_CONTENT = "detectable_content:"

_FORMAT = "detectable_format:"

_MULTITURN = "multi-turn:"

_COMBINATION = "combination:"

_STARTEND = "startend:"

_CHANGE_CASES = "change_case:"

_PUNCTUATION = "punctuation:"

INSTRUCTION_DICT = {
    _KEYWORD + "existence": KeywordChecker,
    _KEYWORD + "frequency": KeywordFrequencyChecker,
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": KeySentenceChecker,
    _KEYWORD + "forbidden_words": ForbiddenWords,
    _KEYWORD + "letter_frequency": LetterFrequencyChecker,
    _LANGUAGE + "response_language": ResponseLanguageChecker,
    _LENGTH + "number_sentences": NumberOfSentences,
    _LENGTH + "number_paragraphs": ParagraphChecker,
    _LENGTH + "number_words": NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": PlaceholderChecker,
    _CONTENT + "postscript": PostscriptChecker,
    _FORMAT + "number_bullet_lists": BulletListChecker,
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": RephraseParagraph,
    _FORMAT + "constrained_response": ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": (HighlightSectionChecker),
    _FORMAT + "multiple_sections": SectionChecker,
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": RephraseChecker,
    _FORMAT + "json_format": JsonFormat,
    _FORMAT + "title": TitleChecker,
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": ConstrainedStartChecker,
    _COMBINATION + "two_responses": TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": RepeatPromptThenAnswer,
    _STARTEND + "end_checker": EndChecker,
    _CHANGE_CASES + "capital_word_frequency": CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": CommaChecker,
    _STARTEND + "quotation": QuotationChecker,
}

INSTRUCTION_CONFLICTS = {
    _KEYWORD + "existence": {_KEYWORD + "existence"},
    _KEYWORD + "frequency": {_KEYWORD + "frequency"},
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": KeySentenceChecker,
    _KEYWORD + "forbidden_words": {_KEYWORD + "forbidden_words"},
    _KEYWORD + "letter_frequency": {_KEYWORD + "letter_frequency"},
    _LANGUAGE + "response_language": {
        _LANGUAGE + "response_language",
        _FORMAT + "multiple_sections",
        _KEYWORD + "existence",
        _KEYWORD + "frequency",
        _KEYWORD + "forbidden_words",
        _STARTEND + "end_checker",
        _CHANGE_CASES + "english_capital",
        _CHANGE_CASES + "english_lowercase",
    },
    _LENGTH + "number_sentences": {_LENGTH + "number_sentences"},
    _LENGTH + "number_paragraphs": {
        _LENGTH + "number_paragraphs",
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_sentences",
        _LENGTH + "nth_paragraph_first_word",
    },
    _LENGTH + "number_words": {_LENGTH + "number_words"},
    _LENGTH + "nth_paragraph_first_word": {
        _LENGTH + "nth_paragraph_first_word",
        _LENGTH + "number_paragraphs",
    },
    _CONTENT + "number_placeholders": {_CONTENT + "number_placeholders"},
    _CONTENT + "postscript": {_CONTENT + "postscript"},
    _FORMAT + "number_bullet_lists": {_FORMAT + "number_bullet_lists"},
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": RephraseParagraph,
    _FORMAT + "constrained_response": set(INSTRUCTION_DICT.keys()),
    _FORMAT + "number_highlighted_sections": {_FORMAT + "number_highlighted_sections"},
    _FORMAT + "multiple_sections": {
        _FORMAT + "multiple_sections",
        _LANGUAGE + "response_language",
        _FORMAT + "number_highlighted_sections",
    },
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": RephraseChecker,
    _FORMAT + "json_format": set(INSTRUCTION_DICT.keys()).difference(
        {_KEYWORD + "forbidden_words", _KEYWORD + "existence"}
    ),
    _FORMAT + "title": {_FORMAT + "title"},
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": ConstrainedStartChecker,
    _COMBINATION + "two_responses": set(INSTRUCTION_DICT.keys()).difference(
        {
            _KEYWORD + "forbidden_words",
            _KEYWORD + "existence",
            _LANGUAGE + "response_language",
            _FORMAT + "title",
            _PUNCTUATION + "no_comma",
        }
    ),
    _COMBINATION + "repeat_prompt": set(INSTRUCTION_DICT.keys()).difference(
        {_KEYWORD + "existence", _FORMAT + "title", _PUNCTUATION + "no_comma"}
    ),
    _STARTEND + "end_checker": {_STARTEND + "end_checker"},
    _CHANGE_CASES + "capital_word_frequency": {
        _CHANGE_CASES + "capital_word_frequency",
        _CHANGE_CASES + "english_lowercase",
        _CHANGE_CASES + "english_capital",
    },
    _CHANGE_CASES + "english_capital": {_CHANGE_CASES + "english_capital"},
    _CHANGE_CASES + "english_lowercase": {
        _CHANGE_CASES + "english_lowercase",
        _CHANGE_CASES + "english_capital",
    },
    _PUNCTUATION + "no_comma": {_PUNCTUATION + "no_comma"},
    _STARTEND + "quotation": {_STARTEND + "quotation", _FORMAT + "title"},
}


def conflict_make(conflicts):
    """Makes sure if A conflicts with B, B will conflict with A.

    Args:
      conflicts: Dictionary of potential conflicts where key is instruction id
        and value is set of instruction ids that it conflicts with.

    Returns:
      Revised version of the dictionary. All instructions conflict with
      themselves. If A conflicts with B, B will conflict with A.
    """
    for key in conflicts:
        for k in conflicts[key]:
            conflicts[k].add(key)
        conflicts[key].add(key)
    return conflicts


##### UTILS.PY

import dataclasses
from typing import Dict, Optional, Union


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
        inp,
        response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
        inp,
        response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"] if "prompt" in doc else doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc
