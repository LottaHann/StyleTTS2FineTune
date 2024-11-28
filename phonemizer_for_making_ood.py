from phonemizer import phonemize

# Full array of sentences (adding to your original examples)
sentences = [
    "It was hard not knowing what was going on. I guess I felt replaced or like I wasn’t important to you anymore.",
    "I was just thinking back to high school and all those late-night talks we had. We were so sure we’d make it big, you know?",
    "Alex! It’s been forever! I can hardly believe we’re actually meeting in person after all this time!",
    "Maybe I should’ve been more understanding, too. We both could’ve handled it better. But it’s good we’re talking now.",
    "I know, it’s surreal. You look almost the same, but maybe with a little more… wisdom, you know? How has life been treating you?",
    "Fine, fine! And don’t worry, this time I’m sticking around. I’ve missed having you in my life more than you know.",
    "Yeah, we thought we’d conquer the world together. I miss that confidence. Life turned out so differently than I imagined.",
    "Honestly, it’s been a rollercoaster. Between work and relationships, there’s been a lot to handle, but I’m doing alright.",
    "It really is. I hate that I let so much time pass, but I don’t want to make the same mistake again.",
    "I know, and I’m sorry. I was going through some personal stuff, and I felt so overwhelmed that I didn’t know how to explain it.",
    "Deal! But you’re buying. Consider it your apology for all the unanswered texts.",
    "You were never unimportant to me. I just didn’t handle things right.",
    "I’d like that too. You know, we’re here now, and that’s what matters, right? We’re both here, trying to make things right.",
    "And that hurt more than I want to admit.",
    "It feels strange to be here, looking back.",
    "Seeing you just makes everything feel lighter somehow.",
    "Let’s try to reconnect, start fresh.",
    "I should’ve talked to you instead of letting my issues push me away from everyone, including you.",
    "I can’t believe you remembered my favorite song after all these years.",
    "She gazed out the window, watching the raindrops race down the glass.",
    "Do you ever wonder what might have been if we took that chance?",
    "The old house creaked with memories long forgotten.",
    "It’s never too late to start over, you know.",
    "He sighed deeply, feeling the weight of his choices.",
    "You always knew how to make me laugh, even on my worst days.",
    "The sun set over the horizon, painting the sky in hues of pink and orange.",
    "I thought you’d moved on and left all this behind.",
    "She tucked a stray hair behind her ear, a nervous habit she couldn’t shake.",
    "Why didn’t you tell me the truth from the beginning?",
    "The silence between them was filled with unspoken words.",
    "Sometimes, I feel like I’m just going through the motions.",
    "He shuffled his feet, unsure of how to respond.",
    "Promise me you’ll keep in touch this time.",
    "The city lights sparkled below, a testament to the life bustling beneath.",
    "I never meant to hurt you; that was never my intention.",
    "She smiled softly, memories flooding back.",
    "What’s the one thing you regret the most?",
    "The aroma of fresh coffee filled the air, comforting and familiar.",
    "I think it’s time we finally faced the past.",
    "He looked away, his eyes betraying a hint of sadness.",
    "Are you happy with how things turned out?",
    "The distant sound of laughter echoed through the empty halls.",
    "Maybe we can find a way to make it work this time.",
    "She picked up the old photograph, tracing the edges with her fingertip.",
    "It’s funny how life brings people back together.",
    "The leaves rustled in the gentle breeze, whispering secrets.",
    "I’ve been meaning to tell you something important.",
    "He took a deep breath, gathering his thoughts.",
    "Do you remember the promise we made under the stars?",
    "The warmth of the fireplace enveloped them, warding off the chill.",
    "I can’t keep pretending that everything is okay.",
    "She glanced at her watch, realizing how late it had become.",
    "There’s so much I wish I could change.",
    "The melody of the piano drifted softly through the room.",
    "You were always the brave one between us.",
    "He chuckled, shaking his head at the memory.",
    "What if we just took a chance, right here and now?",
    "The scent of blooming roses filled the garden.",
    "I wrote you letters, but I never had the courage to send them.",
    "She looked up at the night sky, the stars twinkling above.",
    "Maybe some things are better left unsaid.",
    "The distant thunder hinted at an approaching storm.",
    "I’ve missed talking to you like this.",
    "He reached out, hesitating before letting his hand fall.",
    "We were so young and naïve back then.",
    "The echo of footsteps faded into the darkness.",
    "Do you think people can really change?",
    "She pondered the question, lost in thought.",
    "I believe everyone deserves a second chance.",
    "The waves crashed against the shore, a rhythmic lullaby.",
    "Sometimes, I wish I could go back and do it all differently.",
    "He nodded slowly, understanding her sentiment.",
    "Life doesn’t give us do-overs, but it does offer new beginnings.",
    "The fragrance of freshly baked bread wafted from the kitchen.",
    "You always knew how to find the silver lining.",
    "She laughed, a sound that warmed his heart.",
    "Well, someone has to stay optimistic.",
    "The first drops of rain began to fall, tapping gently on the roof.",
    "We should probably head inside before we get soaked.",
    "He offered his hand, helping her up.",
    "Thank you for meeting me tonight.",
    "She smiled, her eyes reflecting the glow of the streetlights.",
    "It was long overdue, don’t you think?",
    "The familiar path led them back to where it all began.",
    "Do you remember when we used to race down this hill?",
    "He felt a pang of disappointment at her words.",
    "Will I see you again soon?",
    "She paused, considering her response.",
    "I’d like that very much.",
    "The gentle hum of the city at night surrounded them.",
    "Take care of yourself, okay?",
    "He watched as she walked away, fading into the night.",
    "Wait! he called out impulsively.",
    "She turned, hope flickering in her eyes.",
    "There’s something else I need to tell you.",
    "The wind picked up, swirling leaves around their feet.",
    "What is it? she asked softly.",
    "He took a step closer, his heart pounding.",
    "I never stopped thinking about you.",
    "Silence hung between them, heavy with meaning.",
    "I thought I was the only one who felt that way.",
    "Relief washed over him at her confession.",
    "Maybe we owe it to ourselves to see where this goes.",
    "She nodded, a smile spreading across her face.",
    "I’d like that chance.",
    "Hey man, did you catch the latest episode of that sci-fi series—it’s mind-blowing!",
    "Alas, poor Yorick! I knew him, Horatio—a fellow of infinite jest...",
    "The algorithm’s complexity grew exponentially—time and space became scarce resources.",
    "Wait—you’re telling me you hacked into the mainframe in under ten minutes?",
    "By my troth, this banquet is the grandest I’ve ever beheld!",
    "The drone hovered silently above, its sensors scanning the terrain—searching for any signs of life.",
    "Dude, that’s so lit! We totally have to try that new VR game.",
    "Henceforth, I shall not partake in such folly—my honor forbids it.",
    "The spaceship’s engines roared to life—a journey across galaxies awaited them.",
    "You’ve got mail—an old-school phrase, but it still makes me smile.",
    "The knight drew his sword—ready to defend the kingdom to his last breath.",
    "System error 404—not found, the screen flashed ominously.",
    "OMG, can you believe she actually said that? Like, no way!",
    "The ancient scroll revealed secrets long forgotten—whispers of a lost civilization.",
    "Captain, we’ve reached warp speed—the stars are ours to explore!",
    "Verily, I say unto thee, the path is fraught with peril.",
    "The hacker typed furiously—code streaming across the monitor.",
    "Bruh, that concert was epic—best night ever!",
    "The old clock tower struck midnight—a haunting chime echoed through the deserted streets.",
    "Initializing sequence—stand by for launch in T-minus ten seconds.",
    "Pray tell, what brings thee to these parts?",
    "The robotic assistant beeped cheerfully—How may I assist you today?",
    "Y’all coming to the barbecue this weekend? It’s gonna be a hoot!",
    "The alchemist’s laboratory was filled with strange concoctions—bubbling and steaming.",
    "Eureka! I’ve found the solution at last!",
    "The city skyline was a digital masterpiece—neon lights and holograms everywhere.",
    "She sells seashells by the seashore—try saying that three times fast!",
    "The samurai stood silently—his blade reflecting the moonlight.",
    "Upload complete—data transfer successful.",
    "Gadzooks! That was a close call!",
    "The AI became self-aware—an event that would change humanity forever.",
    "Sup? Haven’t seen you around here before.",
    "The pirate’s flag fluttered in the wind—a skull and crossbones emblazoned on black cloth.",
    "Reboot the system—we’re running out of time!",
    "To infinity—and beyond! he shouted, leaping into action.",
    "The steam engine chugged along—the heartbeat of the industrial revolution.",
    "Could you fax that over? Wait—do we even have a fax machine anymore?",
    "The virtual reality simulation blurred the lines between real and artificial.",
    "Forsooth, the queen shall grace us with her presence anon.",
    "The quantum computer processed calculations beyond human comprehension.",
    "LOL, that meme you sent me was hilarious!",
    "The detective examined the clue—a single strand of crimson thread.",
    "All systems operational—prepare for hyperspace jump.",
    "I beseech thee, grant me but one more chance.",
    "The meteor streaked across the sky—a harbinger of change.",
    "Hashtag blessed, she said with a wink.",
    "The librarian whispered, Shhh… this is a place of knowledge.",
    "Downloading update—please do not turn off your device.",
    "The warrior’s battle cry echoed—resonating with courage and defiance.",
    "And so, the story comes to an end—but every end is a new beginning..."
]


# Phonemize the sentences and save them to a file
phonemized_sentences = [
    f"{phonemize(sentence, language='en-us', backend='espeak', strip=True)} |1"
    for sentence in sentences
]

# Save to a text file
with open("phonemized_sentences.txt", "w") as file:
    file.writelines(f"{line}\n" for line in phonemized_sentences)

print("Phonemized sentences have been saved to 'phonemized_sentences.txt'.")
