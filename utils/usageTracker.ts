// A simple client-side usage tracker using localStorage.
// In a production app, this would be handled by a secure backend and user authentication.

const FREE_TIER_LIMIT = 10; // 2 books (5 pages each)
const PAID_TIER_LIMIT = 400;
const USAGE_COUNT_KEY = 'coloringBookUsageCount';
const PLAN_KEY = 'coloringBookPlan';

/**
 * Checks if the user has an active paid plan.
 */
export const isSubscribed = (): boolean => {
    return window.localStorage.getItem(PLAN_KEY) === 'paid';
};

/**
 * Gets the current number of generated images.
 */
export const getUsageCount = (): number => {
    const count = window.localStorage.getItem(USAGE_COUNT_KEY);
    return count ? parseInt(count, 10) : 0;
};

/**
 * Increments the image generation count.
 */
export const incrementUsageCount = (amount: number): void => {
    const currentCount = getUsageCount();
    window.localStorage.setItem(USAGE_COUNT_KEY, (currentCount + amount).toString());
};

/**
 * Checks if the user has reached their plan's limit.
 */
export const hasReachedLimit = (): boolean => {
    const count = getUsageCount();
    if (isSubscribed()) {
        // This user is on a paid plan
        return count >= PAID_TIER_LIMIT;
    }
    // This user is on the free plan
    return count >= FREE_TIER_LIMIT;
};

/**
 * Activates the paid tier for the user.
 * In a real-world scenario, this would be triggered by a successful payment webhook.
 * For this simulation, we call it when the user clicks to upgrade.
 */
export const activatePaidTier = (): void => {
    window.localStorage.setItem(PLAN_KEY, 'paid');
};
