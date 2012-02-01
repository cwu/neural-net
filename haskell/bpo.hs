import Debug.Trace
import Data.List

type Real' = Float
data Layer = Nil | Layer { weights :: [[Real']], thresholds :: [Real']} deriving (Show)
type NeuralNet = [Layer]

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + 2.7182 ** (-x))

layerOut :: Layer -> [Real'] -> [Real']
layerOut (Layer neurons theta) input =
  let
    activate (i,w) = map (i*) w
    neuronThresholds = map ((-1)*) theta
    neuronOutputs = zipWith (curry activate) input neurons
    stimulus = foldl' (zipWith (+)) neuronThresholds neuronOutputs
  in map sigmoid stimulus

learn :: Real' -> NeuralNet -> [Real'] -> [Real'] -> NeuralNet
learn learnRate net input answer =
  let
    adjustWeights :: [Real'] -> Real' -> [Real'] -> [Real']
    adjustWeights errorGradient i w =
      zipWith (\w g -> w + learnRate * i * g) w errorGradient

    --call with acc = (Nil, input)
    calcOutputs :: NeuralNet -> [(Layer, [Real'])] -> [(Layer, [Real'])]
    calcOutputs [] acc = acc
    calcOutputs (l:ls) acc@((_,i):_) = seq acc $ calcOutputs ls ((l, output) : acc)
                    where output = layerOut l i

    improve :: Layer -> [Real'] -> [Real'] -> [Real'] -> (Layer, [Real'])
    improve (Layer w t) i y e =
      let
        g = zipWith (\y e -> y * (1 - y) * e) y e
        w' = zipWith (adjustWeights g) i w
        t' = adjustWeights g (-1) t
        e' = map (sum . zipWith (*) g) w
      in (Layer w' t', e')

    -- note must be used on calcOutput starting acc with (Nil, input)
    adjust :: [(Layer, [Real'])] -> NeuralNet
    adjust ((l0,y0):xs@((_, i0):_)) =
      let
        adjust' :: [(Layer, [Real'])] -> [Real'] -> NeuralNet -> NeuralNet
        adjust' ((Nil,_):_) _ acc = acc
        adjust' ((l,y):xs@((_,i):_)) e acc = seq acc $ adjust' xs e' (l':acc)
          where (l', e') = improve l i y e

        e0 = zipWith (-) answer y0
        (l0', e1) = improve l0 i0 y0 e0
      in adjust' xs e1 [l0']

  in adjust $ calcOutputs net [(Nil, input)]

train :: NeuralNet -> [[Real']] -> [[Real']] -> Real' -> Int -> (NeuralNet, Int)
train net inputs answers learnRate epoch =
  let
    smarter = foldl' ((\f x (y,z) -> f x y z) $ learn learnRate) net (zip inputs answers)
    --ineffecient error finding
    errors = [zipWith (-) (netOut smarter i) a | (i,a) <- zip inputs answers]
    sumSqrErrs = (sum . map (sum . map (^ 2))) errors
  in if sumSqrErrs > 0.001 then train smarter inputs answers learnRate (epoch+1)
    else (smarter, epoch)

-- 56395

netOut :: NeuralNet -> [Real'] -> [Real']
netOut net input = foldl' (flip layerOut) input net


--------------------------------------------------------------------------------
-- trace (show smarter ++ "\nerr: " ++ show sumSqrErrs ++ "  | epoch: " ++ show x ++ "\n\n") $
-- traincycle dumb learningRate = foldl ((\f a x (y,z)-> f x y z a) learn learningRate) dumb (zip input out)

net = [(Layer [[0.5,0.9],[0.4,1.0]] [0.8,-0.1]), (Layer [[-1.2],[1.1]] [0.3])] :: NeuralNet
smarter = learn 0.1 net [1.0,1.0] [0.0]
input = [[1,1],[0,1],[1,0],[0,0]] :: [[Real']]
out = [[0],[1],[1],[0]] :: [[Real']]

test = train net input out 0.1 0

main = print test
