// Fix: Corrected the import path for GoogleGenAI and Type.
import { GoogleGenAI, Type } from "@google/genai";
import { Chat } from "@google/genai";

// Ensure the API key is available
if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable is not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const chatSessions: { [key: string]: Chat } = {};

/**
 * Generates 5 distinct coloring page images based on a theme and language.
 * It first generates ideas in the specified language and then creates an image for each idea.
 */
export const generateColoringPages = async (theme: string, languageName: string): Promise<string[]> => {
    try {
        // Step 1: Generate 5 distinct ideas for coloring pages based on the theme.
        const ideaGenerationResponse = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `List exactly 5 simple, distinct, and creative scenes for a children's coloring book based on the theme "${theme}". Each scene should be a short descriptive phrase. Please provide the list in ${languageName}.`,
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        scenes: {
                            type: Type.ARRAY,
                            items: {
                                type: Type.STRING,
                                description: "A single coloring page scene description."
                            }
                        }
                    }
                },
            },
        });

        const ideaJson = JSON.parse(ideaGenerationResponse.text);
        const scenes: string[] = ideaJson.scenes;

        if (!scenes || scenes.length === 0) {
            throw new Error("Could not generate coloring page ideas.");
        }

        // Step 2: Generate an image for each scene.
        const imagePromises = scenes.slice(0, 5).map(scene => {
            // The prompt for the image model itself should be in English for best performance,
            // but the 'scene' variable will contain the description in the user's chosen language.
            const prompt = `A children's coloring book page of ${scene}. It should be simple black and white line art, with thick, bold outlines, no shading, and no gray areas. The style must be clean, charming, and very easy for a young child to color. White background.`;
            
            return ai.models.generateImages({
                model: 'imagen-4.0-generate-001',
                prompt: prompt,
                config: {
                    numberOfImages: 1,
                    outputMimeType: 'image/png',
                    aspectRatio: '4:3',
                },
            });
        });

        const imageResults = await Promise.all(imagePromises);
        
        return imageResults.map(res => {
            if (res.generatedImages && res.generatedImages.length > 0) {
                return res.generatedImages[0].image.imageBytes;
            }
            throw new Error("Image generation failed for one or more prompts.");
        });

    } catch (error) {
        console.error("Error generating coloring pages:", error);
        throw new Error("Failed to generate coloring book. Please try again.");
    }
};

/**
 * Gets a response from the Gemini chat model in a specific language.
 */
export const getChatResponse = async (sessionId: string, message: string, languageName: string): Promise<string> => {
    try {
        const sessionKey = `${sessionId}-${languageName}`;
        if (!chatSessions[sessionKey]) {
            chatSessions[sessionKey] = ai.chats.create({
                model: 'gemini-2.5-flash',
                config: {
                    systemInstruction: `You are a friendly and helpful assistant for parents and kids using a coloring book app. Keep your answers concise and cheerful. Please respond ONLY in ${languageName}.`,
                },
            });
        }
        
        const chat = chatSessions[sessionKey];
        const result = await chat.sendMessage({ message });
        return result.text;
    } catch (error) {
        console.error("Error getting chat response:", error);
        throw new Error("I'm having a little trouble thinking right now. Please try again in a moment!");
    }
};