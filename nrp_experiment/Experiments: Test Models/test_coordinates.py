# Imported Python Transfer Function
@MapVariable("coordinates",initial_value = (160.,120.),scope = nrp.GLOBAL) 
@nrp.Robot2Neuron()
def test_coordinates(t,coordinates):
    coordinates.value = (160.,120.) 
    return
    time = t % 20  
    if time > 15: 
        coordinates.value = (130.,200.) 
    if time > 10:
        coordinates.value = (250.,200.) 
        return 
    if time > 5:
        coordinates.value = (250,80.) 
        return 
    if time > 0:
        coordinates.value = (130,80)
        return
