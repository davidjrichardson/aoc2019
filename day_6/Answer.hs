import Input
import qualified Data.Tree as Tree
import Data.List.Split

getChildren :: String -> [String]
getChildren parent = (map (snd)) . (filter (\a -> (fst a) == parent)) $ pairs

-- Question 1
tree = Tree.unfoldTree (\x -> (x, getChildren x)) $ "COM"
result_1 = (sum) . (map (\a -> fst a * (length $ snd a))) $ zip [0..] $ Tree.levels $ tree

-- Question 2