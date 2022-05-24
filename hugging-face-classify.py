from transformers import pipeline

text = """
Elon Musk said twice this week that he plans to vote for Republicans in upcoming elections, even though he says he previously voted for Democrats such as former President Barack Obama.

Musk’s stated political leanings will not surprise people who follow the celebrity CEO’s interactions and proclamations on Twitter and elsewhere.

Behind the scenes, Musk and his biggest companies, SpaceX and Tesla, have worked to influence the U.S. political landscape for years through lobbying and political donations. Combined, SpaceX and Tesla spent over $2 million on lobbying in 2021. They tend to spend on both sides of the aisle.

But Musk has been historically anti-union, opposed to a billionaire’s tax and is a vocal critic of President Joe Biden.

Meanwhile, Texas Gov. Greg Abbott, a Republican, said in a September interview with CNBC that Musk approved of his red-state social policies, which have included severe abortion restrictions, book bans that called for LGBTQ memoirs to be removed from school curriculum or libraries, and abuse investigations into families pursuing gender-affirming care for transgender children.

On Wednesday, Musk wrote on Twitter:

“In the past I voted Democrat, because they were (mostly) the kindness party. But they have become the party of division & hate, so I can no longer support them and will vote Republican. Now, watch their dirty tricks campaign against me unfold...” adding a movie popcorn emoji for emphasis.


The tweet followed earlier statements at the All In Summit in Miami on Monday, where Elon Musk accused Twitter of having a strong left-wing bias, saying during a podcast recording, “I would classify myself as a moderate, neither Republican or Democrat. In fact, I have voted overwhelmingly for Democrats historically. Overwhelmingly. I might never have voted Republican. Now, this election? I will.”

Musk has characterized his pending $44 billion acquisition of Twitter a “moderate takeover” of the platform, not a right-wing takeover. But he then proceeded to bash the Democratic Party.

Spurred by podcast and event host Jason Calacanis, who is raising funds to help Musk acquire Twitter, the Tesla CEO said, “The Democratic party is overly controlled by the unions and the trial lawyers, particularly the class-action lawyers.”

On the social platform, Musk has frequently insulted and scrapped with elected Democrats, including Biden, Sens. Elizabeth Warren and Ron Wyden, and Rep. Alexandria Ocasio-Cortez.

By contrast, he tends to engage in a friendly and nonconfrontational manner with right-wing elected officials like GOP Rep. Lauren Boebert and far-right personalities including Steven Crowder, Dinesh D’Souza and others.

Crowder, a podcast host who bills himself as a comedian, was once suspended from YouTube for violating the platform’s hate speech policy after he made comments against trans people. He was also denounced by the Asian American Journalists Association after he made remarks about a broadcast journalist at San Francisco’s KPIX which they deemed racist and sexist.

D’Souza, a conservative commentator, has produced videos and books containing hyperbolic criticism of Democratic leaders for years. He pleaded guilty in 2014 to breaking campaign finance laws, and was later pardoned by President Donald Trump. D’Souza was uninvited from the annual conservative CPAC conference in 2018 after he ridiculed survivors of the Parkland, Florida school shooting.

Musk also has said that he would reinstate Trump on Twitter.

Twitter permanently suspended the former president from the platform in January 2021 following an attack by his supporters on the U.S. Capitol. The company said it made the decision following the Jan. 6 riot “due to the risk of further incitement of violence.”
"""

classifier = pipeline("zero-shot-classification")
result = classifier(
    text,
    candidate_labels=["money", "politics", "scandal"],
)

print(result)